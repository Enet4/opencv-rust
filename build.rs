extern crate pkg_config;
extern crate glob;
extern crate gcc;

use glob::glob;
use std::process::Command;
use std::path::{ PathBuf };
use std::fs;
use std::fs::{ File, read_dir };
use std::ffi::OsString;
use std::io::Write;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=gen_rust.py");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let opencv = pkg_config::Config::new().find("opencv").unwrap();
    let mut search_paths = opencv.include_paths.clone();
    search_paths.push(PathBuf::from("/usr/include"));
    let search_opencv = search_paths.iter().map( |p| {
        let mut path = PathBuf::from(p);
        path.push("opencv2");
        path
    }).find( { |path| read_dir(path).is_ok() });
    let actual_opencv = search_opencv
        .expect("Could not find opencv2 dir in pkg-config includes");

    // add 3rdparty lib dit. pkgconfig forgets it somehow.
    let third_party_dir = format!("{}/share/OpenCV/3rdparty/lib", pkg_config::Config::get_variable("opencv", "prefix").unwrap());
    println!("cargo:rustc-link-search=native={}", third_party_dir);

    println!("OpenCV lives in {:?}", actual_opencv);
    println!("Generating code in {:?}", out_dir);

    let mut gcc = gcc::Config::new();
    gcc.flag("-std=c++0x");
    for path in opencv.include_paths {
        gcc.include(path);
    }

    for entry in glob(&(out_dir.clone() + "/*")).unwrap() {
        fs::remove_file(entry.unwrap()).unwrap()
    }

    let opencv_path_as_string = actual_opencv.to_str().unwrap().to_string();
    let modules = glob(&(opencv_path_as_string.clone()+"/*.hpp")).unwrap().map(|entry| {
        let entry = entry.unwrap();
        let mut files = vec!(entry.to_str().unwrap().to_string());

        let module = entry.file_stem().unwrap().to_str().unwrap().to_string();

        files.extend(glob(&(opencv_path_as_string.clone()+"/"+&*module+"/**/*.h*")).unwrap()
            .map(|file| file.unwrap().to_str().unwrap().to_string())
            .filter(|f|
                    !f.ends_with("cvstd.hpp")
                &&  !f.ends_with("cvstd.inl.hpp")
                &&  !f.ends_with("ocl.hpp")
                &&  !f.ends_with("operations.hpp")
                &&  !f.ends_with("persistence.hpp")
                &&  !f.ends_with("ippasync.hpp")
                &&  !f.ends_with("core/ocl_genbase.hpp")
                &&  !f.ends_with("core/opengl.hpp")
                &&  !f.ends_with("core/mat.inl.hpp")
                &&  !f.ends_with("core/hal/intrin_cpp.hpp")
                &&  !f.ends_with("core/hal/intrin_sse.hpp")
                &&  !f.ends_with("core/hal/intrin_neon.hpp")
                &&  !f.ends_with("ios.h")
                &&  !f.contains("cuda")
                &&  !f.contains("eigen")
                &&  !f.contains("/detail/")
                &&  !f.contains("private")
                &&  !f.contains("/superres/")
            ));
        (module, files)
    })
    .filter(|module|    module.0 != "flann"
                    &&  module.0 != "opencv_modules"
                    &&  module.0 != "opencv"
                    &&  module.0 != "ippicv"
                    &&  module.0 != "hal"
    )
    .collect::<Vec<(String,Vec<String>)>>();

    let mut types = PathBuf::from(&out_dir);
    types.push("common_opencv.h");
    {
        let mut types = File::create(types).unwrap();
        for ref m in modules.iter() {
            write!(&mut types, "#include <opencv2/{}.hpp>\n", &*m.0).unwrap();
        }
    }

    let mut types = PathBuf::from(&out_dir);
    types.push("types.h");
    {
        let mut types = File::create(types).unwrap();
        write!(&mut types, "#include <cstddef>\n").unwrap();
    }

    for ref module in modules.iter() {
        let mut cpp = PathBuf::from(&out_dir);
        cpp.push(&*module.0);
        cpp.set_extension("cpp");

        if !Command::new("python")
                            .args(&["gen_rust.py", "hdr_parser.py", &*out_dir, &*module.0])
                            .args(&(module.1.iter().map(|p| {
                                let mut path = actual_opencv.clone();
                                path.push(p);
                                path.into_os_string()
                            }).collect::<Vec<OsString>>()[..]))
                           .status().unwrap().success() {
            panic!();
        }

        gcc.file(cpp);
    }

    let mut return_types = PathBuf::from(&out_dir);
    return_types.push("return_types.h");
    let mut hub_return_types = File::create(return_types).unwrap();
    for entry in glob(&(out_dir.clone() + "/cv_return_value_*.type.h")).unwrap() {
        writeln!(&mut hub_return_types, r#"#include "{}""#,
            entry.unwrap().file_name().unwrap().to_str().unwrap()).unwrap();
    }

    for entry in glob("native/*.cpp").unwrap() {
        gcc.file(entry.unwrap());
    }
    for entry in glob(&(out_dir.clone() + "/*.type.cpp")).unwrap() {
        gcc.file(entry.unwrap());
    }

    gcc.cpp(true).include(".").include(&out_dir)
        .flag("-Wno-c++11-extensions");

    gcc.compile("libocvrs.a");

    for ref module in &modules {
        let e = Command::new("sh").current_dir(&out_dir).arg("-c").arg(
            format!("g++ {}.consts.cpp -o {}.consts `pkg-config --cflags --libs opencv` -L`pkg-config --variable=prefix opencv`/share/OpenCV/3rdparty/lib",
                module.0, module.0)
        ).status().unwrap();
        assert!(e.success());
        let e = Command::new("sh").current_dir(&out_dir).arg("-c").arg(
            format!("./{}.consts > {}.consts.rs", module.0, module.0)
        ).status().unwrap();
        assert!(e.success());
    }

    let mut hub_filename = PathBuf::from(&out_dir);
    hub_filename.push("hub.rs");
    {
        let mut hub = File::create(hub_filename).unwrap();
        for ref module in &modules {
            writeln!(&mut hub, r#"pub mod {};"#, module.0).unwrap();
        }
        writeln!(&mut hub, r#"pub mod types {{"#).unwrap();
        writeln!(&mut hub, "  use libc::{{ c_void, c_char, size_t }};").unwrap();
        for entry in glob(&(out_dir.clone() + "/*.type.rs")).unwrap() {
            writeln!(&mut hub, r#"  include!(concat!(env!("OUT_DIR"), "/{}"));"#,
                entry.unwrap().file_name().unwrap().to_str().unwrap()).unwrap();
        }
        writeln!(&mut hub, r#"}}"#).unwrap();
        writeln!(&mut hub, "#[doc(hidden)] pub mod sys {{").unwrap();
        writeln!(&mut hub, "  use libc::{{ c_void, c_char, size_t }};").unwrap();
        for entry in glob(&(out_dir.clone() + "/*.rv.rs")).unwrap() {
            writeln!(&mut hub, r#"  include!(concat!(env!("OUT_DIR"), "/{}"));"#,
                entry.unwrap().file_name().unwrap().to_str().unwrap()).unwrap();
        }
        for ref module in &modules {
            writeln!(&mut hub, r#"  include!(concat!(env!("OUT_DIR"), "/{}.externs.rs"));"#, module.0).unwrap();
        }
        writeln!(&mut hub, "}}\n").unwrap();
    }
    println!("cargo:rustc-link-lib=ocvrs");

    // now, this is a nightmare.
    // opencv will embark these as .a when they are not available, or
    // use the one from the system
    // and some may appear in one or more variant (-llibtiff or -ltiff, depending on the system)
    fn lookup_lib(third_party_dir:&str, search: &str) {
        for prefix in vec![ "lib", "liblib" ] {
            for path in vec![ third_party_dir, "/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu/" ] {
                for ext in vec!(".a", ".dylib", ".so") {
                    let name = prefix.to_string() + search;
                    let filename = path.to_string() + "/" + &*name + ext;
                    if fs::metadata(filename.clone()).is_ok() {
                        println!("cargo:rustc-link-lib={}", &name[3..]);
                        return
                    }
                }
            }
        }
    }

    lookup_lib(&*third_party_dir, "IlmImf");
    lookup_lib(&*third_party_dir, "tiff");
    lookup_lib(&*third_party_dir, "ippicv");
    lookup_lib(&*third_party_dir, "jpeg");
    lookup_lib(&*third_party_dir, "png");
    lookup_lib(&*third_party_dir, "jasper");
    lookup_lib(&*third_party_dir, "webp");

    println!("cargo:rustc-link-lib=z");
}

