library(rbenchmark)
library(BDgraph)
library(BGSL)

# rgwish p = 4 test --------------------------------------------------------------
p = 4
b = 103
G = matrix(c(1,1,1,0,
             1,1,0,1,
             1,0,1,1,
             0,1,1,1), nrow = p, ncol = p, byrow = T)
D = matrix(c(136.431, -10.15, 8.027, 2.508,
             -10.15, 93.417, -2.122, -16.162,
             8.027, -2.122, 116.652, 11.62,
             2.508, -16.162, 11.62, 120.203), nrow = p, ncol = p, byrow = T)

test_rgwish_p4 =
  benchmark(
    "BGSL" = {
      K_BGSL = rGwish(G = G, b = b, D = D)$Matrix
    },
    "BDgraph" = {
      K_BDgraph = BDgraph::rgwish(n=1, adj = G, b = b, D = D)
    },
    replications = 1000,
    columns = c("test", "replications", "elapsed",
                "relative", "user.self", "sys.self")
  )
test_rgwish_p4


# IG p = 6 ----------------------------------------------------------------
p = 6
b = 3
D = diag(p)
G = matrix(c(1,1,1,1,0,0,
             0,1,1,1,0,0,
             0,0,1,1,1,1,
             0,0,0,1,1,1,
             0,0,0,0,1,0,
             0,0,0,0,0,1), nrow = p, ncol = p, byrow = T)
n = 100
U = matrix(c( 77.084, -8.831, -39.253, -31.959,  -0.429,  -8.289,
              -8.831, 5.597, 4.514, 3.514, 0.721, 1.150,
              -39.253, 4.514,  35.598,   8.963,  13.194,  -5.765,
              -31.959,  3.514, 8.963,  40.763, -20.965,  25.707,
              -0.429,  0.721,  13.194, -20.965,  46.365, -22.246,
              -8.289,  1.150, -5.765, 25.707,-22.246,  40.247), nrow = 6, ncol = 6, byrow = T)

test_IGpost_p6 =
  benchmark(
    "BGSL" = {
      K_BGSL = log_Gconstant(G, b = b+n, D = D+U, MCiteration = 1000)
    },
    "BDgraph" = {
      K_BDgraph = BDgraph::gnorm(G, b = b+n, D = D+U, iter = 1000)
    },
    replications = 1000,
    columns = c("test", "replications", "elapsed",
                "relative", "user.self", "sys.self")
  )
test_IGpost_p6

