#include <string.h>
#include <stdio.h>
#include <math.h>			/* for findk */

#if defined(_WIN32)
#include <windows.h>
#endif

#include "pdefs.h"

#ifdef __STDC__
#include <stdlib.h>
#endif
#include "precision.h"
#include "pfactor.h"

#ifdef __STDC__
extern unsigned *pfactorbase(precision n, unsigned k,
   unsigned *m, unsigned aborts);
extern double  pomeranceLpow(double n, double alpha);
#else
extern unsigned *pfactorbase();
extern double  pomeranceLpow();
#endif

int verbose = 0;
int debug = 0;

extern unsigned cfracNabort;
extern unsigned cfracTsolns;
extern unsigned cfracPsolns;
extern unsigned cfracT2solns;
extern unsigned cfracFsolns;


extern unsigned short primes[];
extern unsigned primesize;

/*
 * Return the value of "f(p,d)" from Knuth's exercise 28
 */
float pfKnuthEx28(p, d)
   unsigned p;
   precision d;
{
   register float res;
   precision k = pUndef;

   (void) pparm(d);
   if (p == 2) {
      if (peven(d)) {
	 pset(&k, phalf(d));
	 if (peven(k)) {
	    res = 2.0/3.0 + pfKnuthEx28(2,k)/2.0;    /* eliminate powers of 2 */
	 } else {			      /* until only one 2 left in d. */
	    res = 1.0/3.0;           /* independent of (the now odd) k. Wow! */
	 }
      } else {				/* d now odd */
	 pset(&k, phalf(d));
	 if (podd(k)) {
	    res = 1.0/3.0;		/* f(2,4k+3): d%8 == 3 or 7 */
	 } else {
	    if (podd(phalf(k))) {
	       res = 2.0/3.0;		/* f(2,8k+5): d%8 == 5 */
	    } else {
	       res = 4.0/3.0; 		/* f(2,8k+1): d%8 == 1 */
	    }
	 }
      }
   } else {		/* PART 3: p odd, d could still be even (OK) */
      pset(&k, utop(p));
      if peq(ppowmod(d, phalf(psub(k, pone)), k), pone) {
	 res = (float) (p+p) / (((float) p)*p-1.0);  /* beware int overflow! */
      } else {
	 res = 0.0;
      }
   }

   pdestroy(k);
   pdestroy(d);
   if (debug > 1) {
      fprintf(stdout, "f(%u,", p);
      fprintf(stdout, "d) = %9.7f\n", res);
   }
   return res;
}

float cfrac_logf(unsigned p, precision n, unsigned k)
{
   register float     res;

   (void) pparm(n);

#if 0	/* old code for non-float machines; not worth the cost */
   pset(&r, utop(k));
   log2sqrtk = plogb(pipow(r, q >> 1), ptwo);
   fplog2p = (f(p,pmul(r,n),q) * plogb(pipow(utop(p),q),ptwo)+(q>>1))/q;
#endif

   res  = pfKnuthEx28(p, pmul(itop(k),n)) * log((double) p);
   /* res -= log((double) k) * 0.5; */

   pdestroy(n);
   return res;
}

/*
 * Find the best value of k for the given n and m.
 *
 * Input/Output:
 *    n      - the number to factor
 *    m      - pointer to size of factorbase (0 = select "best" size)
 *    aborts - the number of early aborts
 */
unsigned findk(n, m, aborts, maxk)
   precision n;
   register unsigned *m;
   unsigned aborts, maxk;
{
   unsigned k, bestk = 0, count, bestcount = 0, maxpm;
   float sum, max = -1.0E+15;		/* should be small enough */
   unsigned *p;
   register unsigned i;
   register unsigned short *primePtr;

   (void) pparm(n);

   for (k = 1; k < maxk; k++) {		/* maxk should best be m+m? */
      if (debug) {
	 fputs("kN = ", stdout);
	 fputp(stdout, pmul(utop(k), n)); putc('\n', stdout);
      }
      count = *m;
      p = pfactorbase(n, k, &count, aborts);
      if (p == (unsigned *) 0) {
	 fprintf(stderr, "couldn't compute factor base in findk\n");
	 exit(1);
      }

      maxpm = p[count-1];

      sum = 0.0;
      primePtr = primes;
      while (*primePtr <= maxpm) {
	 sum += cfrac_logf((unsigned) *primePtr++, n, k);
      }
      sum -= log((double) k) * 0.5;
      if (verbose > 2) fprintf(stdout, "%u: %5.2f", k, sum);
      if (debug)  fprintf(stdout, " log(k)/2=%5.2f", log((double) k) * 0.5);
      if (verbose > 2) {
	 fputs("\n", stdout);
	 fflush(stdout);
      }
      if (sum > max) {
	 max       = sum;
	 bestk     = k;
	 bestcount = count;
      }
#ifndef IGNOREFREE
      free(p);
#endif
   }

   *m = bestcount;
   pdestroy(n);
   return bestk;
}

#if defined(BDWGC)
#include "gc.h"
#include "gc/gc_mark.h"

unsigned int GC_count = 0;
static void signal_gc()
{
  GC_count++;
}
#endif

#if defined(BM_LOGFILE)
# include "bench_harness.h"
#endif



extern char *optarg;
extern int optind;

char *progName;

extern int getopt();

#if !defined(BENCHLIB)
int main(argc, argv)
   int argc;
   char *argv[];
#else
int
bench_main (argc, argv)
   int argc;
   char *argv[];
#endif
{
   unsigned m = 0, k = 0;
   unsigned maxCount = 1<<30, count, maxk = 0;
   int ch;
   precision n = pUndef, f = pUndef;
   unsigned aborts = 3;
   unsigned *p;
   double d;

#if defined(BDWGC)
   GC_INIT();
   if (!GC_get_start_callback()) {
      GC_set_start_callback(signal_gc);
      GC_start_performance_measurement();
   } else {
      printf("[%s:%u] | GC-notify callback already set\n", __FUNCTION__, __LINE__);
   }
#endif


   progName = *argv;

   while ((ch = getopt(argc, argv, "a:k:i:dv")) != EOF) switch (ch) {
   case 'a':
      aborts = atoi(optarg);
      break;
   case 'k':
      maxk = atoi(optarg);
      break;
   case 'i':
      maxCount = atoi(optarg);
      break;
   case 'd':
      debug++;
      break;
   case 'v':
      verbose++;
      break;
   default:
usage: fprintf(stderr,
  "usage: %s [-dv] [-a aborts ] [-k maxk ] [-i maxCount ] n [[ m ] k ]\n",
         progName);
      return 1;
   }
   argc -= optind;
   argv += optind;

   if (argc == 0) {
     argc = 1;
     static char* argvx[2] = { "17545186520507317056371138836327483792736", NULL };
     argv = argvx;
   }

   if (argc < 1 || argc > 3) goto usage;

   pset(&n, atop(*argv++));  --argc;
   if (argc) { m = atoi(*argv++);  --argc; }
   if (argc) { k = atoi(*argv++);  --argc; }

   if (k == 0) {
      if (maxk == 0) {
	 maxk = m / 2 + 5;
	 if (verbose) fprintf(stdout, "maxk = %u\n", maxk);
      }
      k = findk(n, &m, aborts, maxk);
      if (verbose) {
	 fprintf(stdout, "k = %u\n", k);
      }
   }

   count = maxCount;

   pcfracInit(m, k, aborts);

   pset(&f, pcfrac(n, &count));
   count = maxCount - count;
   if (verbose) {
      putc('\n', stdout);
      fprintf(stdout, "Iterations     : %u\n", count);
      fprintf(stdout, "Early Aborts   : %u\n", cfracNabort);
      fprintf(stdout, "Total Partials : %u\n", cfracTsolns);
      fprintf(stdout, "Used  Partials : %u\n", cfracT2solns);
      fprintf(stdout, "Full Solutions : %u\n", cfracPsolns);
      fprintf(stdout, "Factor Attempts: %u\n", cfracFsolns);
   }

   if (f != pUndef) {
      fputp(stdout, n);
      fputs(" = ", stdout);
      fputp(stdout, f);
      fputs(" * ", stdout);
      pdivmod(n, f, &n, pNull);
      fputp(stdout, n);
      putc('\n', stdout);
   }

   pdestroy(f);
   pdestroy(n);

   #if defined(BM_LOGFILE)
   #if defined(BDWGC)
   printf("[%s:%d] | number of gc- cycles complete = %u, total-gc-time = %u\n",
                   __FUNCTION__, __LINE__, GC_count, GC_get_full_gc_total_time());
   BM_Harness data = { .bm = "cfrac",
                       .gc_cycles = GC_count,
                       .gc_time_ms = GC_get_full_gc_total_time()};
   #else
   BM_Harness data = { .bm = "cfrac",
                       .gc_cycles = 0,
                       .gc_time_ms = 0};
   #endif
   bmlog(&data);
   #endif

   return 0;
}
