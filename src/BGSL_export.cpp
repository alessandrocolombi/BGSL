#define STRICT_R_HEADERS
#include <Rcpp.h>

//' Multiply a number by two
//' 
//' @param x A single integer.
//' @export
// [[Rcpp::export]]
int timesTwo(int x) {
   return x * 2;
}
