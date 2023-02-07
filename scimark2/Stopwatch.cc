#include <stdlib.h>
#include "Stopwatch.h"
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

double seconds()
{
  struct timespec ts;

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_REALTIME, &ts);
#endif


  return (double)(ts.tv_sec) + (double)(ts.tv_nsec) / 1000000000.0;
}

void Stopwtach_reset(Stopwatch Q)
{
    Q->running = 0;         /* false */
    Q->last_time = 0.0;
    Q->total= 0.0;
}


Stopwatch new_Stopwatch(void)
{
    Stopwatch S = (Stopwatch) aligned_alloc(64,sizeof(Stopwatch_struct));
    if (S == NULL)
        return NULL;

    Stopwtach_reset(S);
    return S;
}

void Stopwatch_delete(Stopwatch S)
{
    if (S != NULL)
        free(S);
}


/* Start resets the timer to 0.0; use resume for continued total */

void Stopwatch_start(Stopwatch Q)
{
    if (! (Q->running)  )
    {
        Q->running = 1;  /* true */
        Q->total = 0.0;
        Q->last_time = seconds();
    }
}
   
/** 
    Resume timing, after stopping.  (Does not wipe out
        accumulated times.)

*/

void Stopwatch_resume(Stopwatch Q)
{
    if (!(Q->running))
    {
        Q-> last_time = seconds(); 
        Q->running = 1;  /*true*/
    }
}
   
void Stopwatch_stop(Stopwatch Q)  
{ 
    if (Q->running) 
    { 
        Q->total += seconds() - Q->last_time; 
        Q->running = 0;  /* false */
    }
}
  
 
double Stopwatch_read(Stopwatch Q)
{  
    
    if (Q->running) 
    {
        double t = seconds();
        Q->total += t - Q->last_time;
        Q->last_time = t;
    }
    return Q->total;
}
        
