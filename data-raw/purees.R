## code to prepare `purees` dataset goes here
data_mat = read.csv("data-raw/purees.csv")
data_mat = data_mat[,-1]
wavelengths <- read.csv("data-raw/wavelengths.csv", sep="")

purees = list("data" = data_mat, "wavelengths" = wavelengths)
usethis::use_data(purees, overwrite = TRUE)
