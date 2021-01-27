#include <pthread.h>
#include <unistd.h>
extern __thread int i;
static int res1, res2;
void thread_main (void *arg) {
  i = (int)arg;
  sleep (1);
  if ((int)arg == 1)
    res1 = (i == (int)arg);
  else
    res2 = (i == (int)arg);
/* I see the concept, but this is incorrect, there is a chance it works, even without TLS */
}
__thread int i;
int main () {
  pthread_t t1, t2;
  i = 5;
  pthread_create (&t1, NULL, thread_main, (void *)1);
  pthread_create (&t2, NULL, thread_main, (void *)2);
  pthread_join (t1, NULL);
  pthread_join (t2, NULL);
  return !(res1 + res2 == 2);
}
