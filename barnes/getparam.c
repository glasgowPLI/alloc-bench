//#line 95 "./null_macros/c.m4.null"

//#line 3 "getparam.C"
/*************************************************************************/
/*                                                                       */
/*  Copyright (c) 1994 Stanford University                               */
/*                                                                       */
/*  All rights reserved.                                                 */
/*                                                                       */
/*  Permission is given to use, copy, and modify this software for any   */
/*  non-commercial purpose as long as this copyright notice is not       */
/*  removed.  All other uses, including redistribution in whole or in    */
/*  part, are forbidden without prior written permission.                */
/*                                                                       */
/*  This software is provided with absolutely no warranty and no         */
/*  support.                                                             */
/*                                                                       */
/*************************************************************************/

/*
 * GETPARAM.C: 
 */
 
#include <string.h>
#include <stdlib.h>
#include "stdinc.h"
#include "getparam.h"
#include "util.h"
#if defined(BDWGC) // @djichthys  - changed to not use deprecated stuff 
#  include "gc.h"
#  define MALLOC GC_MALLOC
#elif defined(BUMPALLOC)
#  define MALLOC bump_alloc
#else 
#  define MALLOC malloc
#endif

local string *defaults = NULL;        /* vector of "name=value" strings */

/*
 * INITPARAM: ignore arg vector, remember defaults.
 */

void initparam(string *argv, string* defv)
  // string *argv, *defv;
{
   defaults = defv;
}

/*
 * GETPARAM: export version prompts user for value.
 */

string getparam(string name)
  // string name;                        /* name of parameter */
{
   // int scanbind(), i, strlen(), leng;
   int i, leng;
   // string extrvalue(), def;
   string def;
   // char buf[128], *strcpy();
   char buf[128];
   char* temp;

   if (defaults == NULL)
      error("getparam: called before initparam\n");
   i = scanbind(defaults, name);
   if (i < 0)
      error("getparam: %s unknown\n", name);
   def = extrvalue(defaults[i]);
   #if !defined(BDWGC) // @djichthys  - changed to not use deprecated stuff 
   //gets(buf); 
   fgets(buf, sizeof(buf), stdin); // @djichthys - replaced previous line with fgets()
   buf[strcspn(buf, "\r\n")] = '\0';
   #else 
   fgets(buf, sizeof(buf), stdin); // @djichthys - replaced previous line with fgets()
   buf[strcspn(buf, "\r\n")] = '\0';
   #endif 
   leng = strlen(buf) + 1;
   if (leng > 1) {
      return (strcpy(MALLOC(leng), buf));
   }
   else {
      return (def);
   }
}

/*
 * GETIPARAM, ..., GETDPARAM: get int, long, bool, or double parameters.
 */

int getiparam(string name)
  // string name;                        /* name of parameter */
{
   // string getparam(), val;
   string val;
   // int atoi();

   for (val = ""; *val == NULL;) {
      val = getparam(name);
   }
   return (atoi(val));
}

long getlparam(string name)
  // string name;                        /* name of parameter */
{
   // string getparam(), val;
   string val;
   // long atol();

   for (val = ""; *val == NULL; )
      val = getparam(name);
   return (atol(val));
}

bool getbparam(string name)
  // string name;                        /* name of parameter */
{
   // string getparam(), val;
  string val;
   for (val = ""; *val == NULL; )
      val = getparam(name);
   if (strchr("tTyY1", *val) != NULL) {
      return (TRUE);
   }
   if (strchr("fFnN0", *val) != NULL) {
      return (FALSE);
   }
   error("getbparam: %s=%s not bool\n", name, val);
   return FALSE;//unreachable code (error exits)
}

double getdparam(string name)
  // string name;                        /* name of parameter */
{
   // string getparam(), val;
   string val;
   // double atof();

   for (val = ""; *val == NULL; ) {
      val = getparam(name);
   }
   return (atof(val));
}



/*
 * SCANBIND: scan binding vector for name, return index.
 */

int scanbind(string bvec[], string name)
  // string bvec[];
  // string name;
{
   int i;
   // bool matchname();

   for (i = 0; bvec[i] != NULL; i++)
      if (matchname(bvec[i], name))
	 return (i);
   return (-1);
}

/*
 * MATCHNAME: determine if "name=value" matches "name".
 */

bool matchname(string bind, string name)
  // string bind, name;
{
   char *bp, *np;

   bp = bind;
   np = name;
   while (*bp == *np) {
      bp++;
      np++;
   }
   return (*bp == '=' && *np == NULL);
}

/*
 * EXTRVALUE: extract value from name=value string.
 */

string extrvalue(string arg)
  // string arg;                        /* string of the form "name=value" */
{
   char *ap;

   ap = (char *) arg;
   while (*ap != NULL)
      if (*ap++ == '=')
	 return ((string) ap);
   return (NULL);
}

