## code to prepare `examples` dataset goes here
load("data-raw/Testp30_exampledata_blocchimozzi_trueval.Rdat")
Trueval_p30 = True_val
load("data-raw/Testp40_exampledata_trueval.Rdat")
Trueval_p40 = True_val
load("data-raw/GGM_p30_blocchimozzi_testJGCS_revision_trueval.Rdat")
Trueval_p30_rep = True_val
load("data-raw/GGM_p40_complete_testJGCS_revision_trueval.Rdat")
Trueval_p40_rep = True_val

examples = list("Trueval_p30" = Trueval_p30,
                "Trueval_p40" = Trueval_p40,
                "Trueval_p30_rep" = Trueval_p30_rep,
                "Trueval_p40_rep" = Trueval_p40_rep)

usethis::use_data(examples, overwrite = TRUE)
