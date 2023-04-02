//+------------------------------------------------------------------+
//|      random values for a given distribution                      |
//+------------------------------------------------------------------+

/*
NOTE: THIS LIBRARY NEEDS SOME WORK !!!
1. THERE ARE SOME BUILT-IN FUNCTIONS IN THE STANDARD LIBRARY THAT MIGHT ALREADY BE BETTER
2. rand() NEEDS PROPER SEEDING
3. IMPLEMENTATIONS FOR VECTORS NEEDED
*/

// random normal distribution
double rand_norm(double mu=0,double sigma=1)
  {
   double random=(double)rand() / RAND_MAX;              // random value within range 0-1
   random/=sqrt(2*M_PI*pow(sigma,2));                 // reduce to the top of the distribution (f(x_val=mu))
   double algsign=1;if (rand()>(0.5*RAND_MAX)){algsign=-1;}    // get random algebraic sign
   return algsign * (mu + sigma * sqrt (-2 * log (random / (1/sqrt(2*M_PI*pow(sigma,2))))));
  }

// random cauchy distribution
double rand_cauchy(double x_peak=0,double gamma=1){
   double random=(double)rand() / RAND_MAX;              // random value within range 0-1
   random/=M_PI*gamma;                                // reduce to the top of the distribution (=f(x_val=x_peak))
   double algsign=1;if (rand()>(0.5*RAND_MAX)){algsign=-1;}    // get random algebraic sign   
   return algsign* (sqrt ( gamma/(random*M_PI) - pow(gamma,2) ) + x_peak);
  }

// random uniform distribution
double rand_uni(double x_mean=0.5,double range=0.5)
  {
   double random=(double)rand() / (0.5*RAND_MAX) - 1;          // random value within range +/- 1
   random*=range;
   random+=x_mean;
   return random;
  }

// random laplace distribution
double rand_laplace(double mu=0,double scale_factor=0.707106781)
  {
   double random=(double)rand() / RAND_MAX;              // random value within range 0-1
   random/=2*scale_factor;                            // reduce to top of distribution (f(x_val=mu)
   double algsign=1;if (rand()>(0.5*RAND_MAX)){algsign=-1;}    // get random algebraic sign
   return mu + algsign*scale_factor*log(random*2*scale_factor);
  }
  
// random pareto distribution
double rand_pareto(double alpha=1,double tail_index=1)
  {
   double random=(double)rand() / RAND_MAX;              // random value within range 0-1
   random*=(alpha*pow(tail_index,alpha))/pow(tail_index,alpha+1); // top of distribution is given for x_val=tail_index
   return pow((alpha*pow(tail_index,alpha))/random,1/(alpha+1));
  }

// random lomax distribution
double rand_lomax(double alpha=1,double tail_index=1)
  {
  double random=(double)rand() / RAND_MAX;              // random value within range 0-1
  random*=(alpha/tail_index)*pow(1/tail_index,-(alpha+1));
  return tail_index*(pow((random*tail_index)/alpha,-1/(alpha+1))-1);
  }

// random binary
ushort rand_bin()
  {
   return (ushort)rand()%2;
  }

// random sign
double rand_sign()
  {
   if (rand()>(0.5*RAND_MAX)){return 1;}
   else {return -1;}
  }

// random boolean
bool rand_bool(){
   if (rand()>(0.5*RAND_MAX)){return true;}
   else {return false;}
  }
