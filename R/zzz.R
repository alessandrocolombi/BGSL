# unload dynamic library when package is detached
.onUnload <- function (libpath) {
  library.dynam.unload("BGSL", libpath)
}
