#include <dlfcn.h>
#include <jni.h>
#include "PapiJ.h"
#include "papi.h"

void *
getPapiFunction(char *f)
{
  void *library, *func;

  library = dlopen("libpapi.so",RTLD_LAZY);
  if(!library) {
#ifdef PAPIJ_DEBUG
    fprintf(stderr,"getPapiFunction: dlopen() failed %s\n",dlerror());
    perror("reason:");
#endif
    return NULL;
  }
  func = dlsym(library, f);
  if(!func) {
#ifdef PAPIJ_DEBUG
    fprintf(stderr,"getPapiFunction: dlsym() failed\n");
    perror("reason:");
#endif
    return NULL;
  }

  return func;
}

JNIEXPORT jint JNICALL Java_PapiJ_flops
  (JNIEnv *env, jobject obj, jobject info)
{

  jclass cls = (*env)->GetObjectClass(env, info);
  jfieldID r_id, p_id, f_id, m_id;

  jfloat real,proc,mflop;
  jlong ins;

  int ret, (*flops)(float *, float *, long_long *, float *);

  if( ! (flops = getPapiFunction("PAPI_flops")) )
    return -1;

  r_id = (*env)->GetFieldID(env, cls, "real_time", "F");
  p_id = (*env)->GetFieldID(env, cls, "proc_time", "F");
  f_id = (*env)->GetFieldID(env, cls, "flpins", "J");
  m_id = (*env)->GetFieldID(env, cls, "mflops", "F");

  if((r_id == 0) || (p_id == 0) || (f_id == 0) || (m_id == 0))
    return -1;

  real =  (*env)->GetFloatField(env, info, r_id);
  proc =  (*env)->GetFloatField(env, info, p_id);
  ins =   (*env)->GetLongField(env, info, f_id);
  mflop = (*env)->GetFloatField(env, info, m_id);

  ret = (*flops)(&real, &proc, &ins, &mflop);
  
  (*env)->SetFloatField(env, info, r_id, real);
  (*env)->SetFloatField(env, info, p_id, proc);
  (*env)->SetLongField(env, info, f_id, ins);
  (*env)->SetFloatField(env, info, m_id, mflop);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_num_1counters
  (JNIEnv *env, jobject obj)
{
  int (*num_counters)(void);

  if( ! (num_counters = getPapiFunction("PAPI_num_counters")) )
    return -1;

  return (*num_counters)();
}

JNIEXPORT jint JNICALL Java_PapiJ_start_1counters
  (JNIEnv *env, jobject obj, jintArray values)
{
  jsize len;
  jint *events;
  int ret, (*start_counters)(int *, int);

  if( ! (start_counters = getPapiFunction("PAPI_start_counters")) )
    return -1;

  len = (*env)->GetArrayLength(env, values);
  events = (*env)->GetIntArrayElements(env, values, 0);

  ret = (*start_counters)((int*)events, len);
  
  (*env)->ReleaseIntArrayElements(env, values, events, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_stop_1counters
  (JNIEnv *env, jobject obj, jlongArray values)
{
  jsize len;
  jlong *events;
  int ret, (*stop_counters)(long_long *, int);

  if( ! (stop_counters = getPapiFunction("PAPI_stop_counters")) )
    return -1;

  len = (*env)->GetArrayLength(env, values);
  events = (*env)->GetLongArrayElements(env, values, 0);

  ret = (*stop_counters)(events, len);

  (*env)->ReleaseLongArrayElements(env, values, events, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_read_1counters
  (JNIEnv *env, jobject obj, jlongArray values)
{
  jsize len;
  jlong *events;
  int ret, (*read_counters)(long_long *, int);

  if( ! (read_counters = getPapiFunction("PAPI_read_counters")) )
    return -1;

  len = (*env)->GetArrayLength(env, values);
  events = (*env)->GetLongArrayElements(env, values, 0);

  ret = (*read_counters)(events, len);

  (*env)->ReleaseLongArrayElements(env, values, events, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_accum_1counters
  (JNIEnv *env, jobject obj, jlongArray values)
{
  jsize len;
  jlong *events;
  int ret, (*accum_counters)(long_long *, int);

  if( ! (accum_counters = getPapiFunction("PAPI_accum_counters")) )
    return -1;

  len = (*env)->GetArrayLength(env, values);
  events = (*env)->GetLongArrayElements(env, values, 0);

  ret = (*accum_counters)(events, len);

  (*env)->ReleaseLongArrayElements(env, values, events, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_accum
  (JNIEnv *env, jobject obj, jobject set, jlongArray values)
{
  jlong *v_arr;
  int ret, (*accum)(int, long_long *), eventSet;
  jfieldID fid;
  jclass class;

  if( ! (accum = getPapiFunction("PAPI_accum")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");
  
  eventSet = (*env)->GetIntField(env, set, fid);

  v_arr = (*env)->GetLongArrayElements(env, values, 0);
  
  ret = (*accum)(eventSet, v_arr);

  (*env)->ReleaseLongArrayElements(env, values, v_arr, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_add_1event
  (JNIEnv *env, jobject obj, jobject set, jint event)
{
  int ret, eventSet, (*add_event)(int, int);   /* JT */
  jfieldID fid;
  jclass class;

  if( ! (add_event = getPapiFunction("PAPI_add_event")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  ret = (*add_event)(eventSet, event);     /* JT */


  (*env)->SetIntField(env, set, fid, eventSet);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_add_1events
  (JNIEnv *env, jobject obj, jobject set, jintArray events)
{
  int num, ret, eventSet, (*add_events)(int, int *, int);   /* JT */
  jint *e_arr;
  jfieldID fid;
  jclass class;

  if( ! (add_events = getPapiFunction("PAPI_add_events")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  e_arr = (*env)->GetIntArrayElements(env, events, 0);
  num = (*env)->GetArrayLength(env, events);

  ret = (*add_events)(eventSet, (int*)e_arr, num);    /* JT */

  (*env)->SetIntField(env, set, fid, eventSet);
  (*env)->ReleaseIntArrayElements(env, events, e_arr, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_cleanup_1eventset
  (JNIEnv *env, jobject obj, jobject set)
{
  int ret, eventSet, (*cleanup_eventset)(int);   /* JT */
  jfieldID fid;
  jclass class;

  if( ! (cleanup_eventset = getPapiFunction("PAPI_cleanup_eventset")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  ret = (*cleanup_eventset)(eventSet);     /* JT */

  (*env)->SetIntField(env, set, fid, eventSet);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_create_1eventset
  (JNIEnv *env, jobject obj, jobject set)
{
  int ret, eventSet, (*create_eventset)(int *);
  jfieldID fid;
  jclass class;

  if( ! (create_eventset = getPapiFunction("PAPI_create_eventset")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  ret = (*create_eventset)(&eventSet);

  (*env)->SetIntField(env, set, fid, eventSet);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_destroy_1eventset
  (JNIEnv *env, jobject obj, jobject set)
{
  int ret, eventSet, (*destroy_eventset)(int *);
  jfieldID fid;
  jclass class;

  if( ! (destroy_eventset = getPapiFunction("PAPI_destroy_eventset")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  ret = (*destroy_eventset)(&eventSet);

  (*env)->SetIntField(env, set, fid, eventSet);

  return ret;
}

JNIEXPORT jobject JNICALL Java_PapiJ_get_1executable_1info
  (JNIEnv *env, jobject obj)
{
  PAPI_exe_info_t *exe, *(*get_executable_info)(void);
  jmethodID mid;
  jclass class;
  jobject exe_obj = NULL;

  if( ! (get_executable_info = getPapiFunction("PAPI_get_executable_info")) )
    return NULL;

  exe = (*get_executable_info)();

  if( ! (class = (*env)->FindClass(env, "PAPI_exe_info")) )
    return NULL;

  mid = (*env)->GetMethodID(env, class, "<init>", 
    "(Ljava/lang/String;Ljava/lang/String;JJJJJJ)V");

  fprintf(stderr, "%s\n", exe->fullname);
  exe_obj = (*env)->NewObject(env, class, mid, 
    (*env)->NewStringUTF(env,exe->fullname), 
    (*env)->NewStringUTF(env,exe->address_info.name),
    (jlong)(jint)(exe->address_info.text_start), (jlong)(jint)(exe->address_info.text_end),
    (jlong)(jint)(exe->address_info.data_start), (jlong)(jint)(exe->address_info.data_end),
    (jlong)(jint)(exe->address_info.bss_start), (jlong)(jint)(exe->address_info.bss_end));

  return exe_obj;
}

JNIEXPORT jobject JNICALL Java_PapiJ_get_1hardware_1info
  (JNIEnv *env, jobject obj)
{
  PAPI_hw_info_t *hw, *(*get_hardware_info)(void);
  jmethodID mid;
  jclass class;
  jobject hw_obj = NULL;

  if( ! (get_hardware_info = getPapiFunction("PAPI_get_hardware_info")) )
    return NULL;

  hw = (*get_hardware_info)();

  if( ! (class = (*env)->FindClass(env, "PAPI_hw_info")) )
    return NULL;

  mid = (*env)->GetMethodID(env, class, "<init>",
    "(IIIILjava/lang/String;ILjava/lang/String;FF)V");

  hw_obj = (*env)->NewObject(env, class, mid, hw->ncpu, hw->nnodes,
    hw->totalcpus, hw->vendor, (*env)->NewStringUTF(env,hw->vendor_string),
    hw->model, (*env)->NewStringUTF(env,hw->model_string), hw->revision,
    hw->mhz);

  return hw_obj;
}

JNIEXPORT jint JNICALL Java_PapiJ_library_1init
  (JNIEnv *env, jobject obj, jint ver)
{
  int (*library_init)(int);

  if( ! (library_init = getPapiFunction("PAPI_library_init")) )
    return -1;

  return (*library_init)(ver); 
}

JNIEXPORT jlong JNICALL Java_PapiJ_get_1real_1cyc
  (JNIEnv *env, jobject obj)
{
  long_long (*get_real_cyc)(void);

  if( ! (get_real_cyc = getPapiFunction("PAPI_get_real_cyc")) )
    return -1;

  return (*get_real_cyc)(); 
}

JNIEXPORT jlong JNICALL Java_PapiJ_get_1real_1usec
  (JNIEnv *env, jobject obj)
{
  long_long (*get_real_usec)(void);

  if( ! (get_real_usec = getPapiFunction("PAPI_get_real_usec")) )
    return -1;

  return (*get_real_usec)(); 
}

JNIEXPORT jlong JNICALL Java_PapiJ_get_1virt_1cyc
  (JNIEnv *env, jobject obj)
{
  u_long_long (*get_virt_cyc)(void);    /* JT */

  if( ! (get_virt_cyc = getPapiFunction("PAPI_get_virt_cyc")) )
    return -1;

  return (*get_virt_cyc)(); 
}

JNIEXPORT jlong JNICALL Java_PapiJ_get_1virt_1usec
  (JNIEnv *env, jobject obj)
{
  u_long_long (*get_virt_usec)(void);     /* JT */

  if( ! (get_virt_usec = getPapiFunction("PAPI_get_virt_usec")) )
    return -1;

  return (*get_virt_usec)(); 
}

JNIEXPORT jint JNICALL Java_PapiJ_list_1events
  (JNIEnv *env, jobject obj, jobject set, jintArray events)
{
  int num, ret, (*list_events)(int, int*, int*), eventSet;
  jint *e_arr;
  jfieldID fid;
  jclass class;

  if( ! (list_events = getPapiFunction("PAPI_list_events")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  num = (*env)->GetArrayLength(env, events);
  e_arr = (*env)->GetIntArrayElements(env, events, 0);

  ret = (*list_events)(eventSet, (int*)e_arr, &num);

  (*env)->ReleaseIntArrayElements(env, events, e_arr, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_perror
  (JNIEnv *env, jobject obj, jint code, jcharArray dest)
{
  int ret, num, (*papi_perror)(int, char *, int);
  jchar *c_arr;

  if( ! (papi_perror = getPapiFunction("PAPI_perror")) )
    return -1;

  num = (*env)->GetArrayLength(env, dest);

  c_arr = (*env)->GetCharArrayElements(env, dest, 0);

  ret = (*papi_perror)(code, (char*)c_arr, num);

  (*env)->ReleaseCharArrayElements(env, dest, c_arr, 0);
  
  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_profil
  (JNIEnv *env, jobject obj, jshortArray buf, jlong offset, jint scale,
      jobject set, jint eventCode, jint thresh, jint flags)
{
  int (*profil)(unsigned short *, unsigned, unsigned long, unsigned, 
                int, int, int, int);
  int num, ret, eventSet;
  jshort *buf_arr;
  jfieldID fid;
  jclass class;

  if( ! (profil = getPapiFunction("PAPI_profil")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  num = (*env)->GetArrayLength(env, buf);

  buf_arr = (*env)->GetShortArrayElements(env, buf, 0);

  ret = (*profil)(buf_arr, num, offset, scale, eventSet, eventCode, 
    thresh, flags);

  (*env)->ReleaseShortArrayElements(env, buf, buf_arr, 0);
  
  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_query_1event
  (JNIEnv *env, jobject obj, jint eventCode)
{
  int (*query_event)(int);

  if( ! (query_event = getPapiFunction("PAPI_query_event")) )
    return -1;

  return (*query_event)(eventCode);
}

JNIEXPORT jint JNICALL Java_PapiJ_read
  (JNIEnv *env, jobject obj, jobject set, jlongArray values)
{
  jlong *v_arr;
  int ret, eventSet, (*papi_read)(int, long_long *);
  jfieldID fid;
  jclass class;

  if( ! (papi_read = getPapiFunction("PAPI_read")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  v_arr = (*env)->GetLongArrayElements(env, values, 0);

  ret = (*papi_read)(eventSet, v_arr);

  (*env)->ReleaseLongArrayElements(env, values, v_arr, 0);

  return ret;
}

JNIEXPORT jint JNICALL Java_PapiJ_reset
  (JNIEnv *env, jobject obj, jobject set)
{
  int eventSet, (*papi_reset)(int);
  jfieldID fid;
  jclass class;

  if( ! (papi_reset = getPapiFunction("PAPI_reset")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  return (*papi_reset)(eventSet);
}

JNIEXPORT jint JNICALL Java_PapiJ_restore
  (JNIEnv *env, jobject obj)
{
  int (*papi_restore)(void);

  if( ! (papi_restore = getPapiFunction("PAPI_restore")) )
    return -1;

  return (*papi_restore)();
}

JNIEXPORT jint JNICALL Java_PapiJ_save
  (JNIEnv *env, jobject obj)
{
  int (*papi_save)(void);

  if( ! (papi_save = getPapiFunction("PAPI_save")) )
    return -1;

  return (*papi_save)();
}

JNIEXPORT jint JNICALL Java_PapiJ_set_1debug
  (JNIEnv *env, jobject obj, jint level)
{
  int (*set_debug)(int);

  if( ! (set_debug = getPapiFunction("PAPI_set_debug")) )
    return -1;

  return (*set_debug)(level);
}

JNIEXPORT jint JNICALL Java_PapiJ_set_1domain
  (JNIEnv *env, jobject obj, jint domain)
{
  int (*set_domain)(int);

  if( ! (set_domain = getPapiFunction("PAPI_set_domain")) )
    return -1;

  return (*set_domain)(domain);
}

JNIEXPORT jint JNICALL Java_PapiJ_set_1granularity
  (JNIEnv *env, jobject obj, jint granularity)
{
  int (*set_granularity)(int);

  if( ! (set_granularity = getPapiFunction("PAPI_set_granularity")) )
    return -1;

  return (*set_granularity)(granularity);
}

JNIEXPORT void JNICALL Java_PapiJ_shutdown
  (JNIEnv *env, jobject obj)
{
  int (*papi_shutdown)(void);

  if( ! (papi_shutdown = getPapiFunction("PAPI_shutdown")) )
    return;

  (*papi_shutdown)();

  return;
}

JNIEXPORT jint JNICALL Java_PapiJ_start
  (JNIEnv *env, jobject obj, jobject set)
{
  int eventSet, (*papi_start)(int);
  jfieldID fid;
  jclass class;

  if( ! (papi_start = getPapiFunction("PAPI_start")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  return (*papi_start)(eventSet);
}

JNIEXPORT jint JNICALL Java_PapiJ_stop
  (JNIEnv *env, jobject obj, jobject set, jlongArray values)
{
  jlong *v_arr;
  int ret, eventSet, (*papi_stop)(int, long_long *);
  jfieldID fid;
  jclass class;

  if( ! (papi_stop = getPapiFunction("PAPI_stop")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  v_arr = (*env)->GetLongArrayElements(env, values, 0);

  ret = (*papi_stop)(eventSet, v_arr);

  (*env)->ReleaseLongArrayElements(env, values, v_arr, 0);

  return ret;
}

JNIEXPORT jstring JNICALL Java_PapiJ_strerror
  (JNIEnv *env, jobject obj, jint code)
{
  char *ret, *(*papi_strerror)(int);

  if( ! (papi_strerror = getPapiFunction("PAPI_strerror")) )
    return NULL;

  ret = (*papi_strerror)(code);

  return (*env)->NewStringUTF(env,ret);
}

JNIEXPORT jint JNICALL Java_PapiJ_write
 (JNIEnv *env, jobject obj, jobject set, jlongArray values)
{
  jlong *v_arr;
  int ret, eventSet, (*papi_write)(int, long_long *);
  jfieldID fid;
  jclass class;

  if( ! (papi_write = getPapiFunction("PAPI_write")) )
    return -1;

  class = (*env)->GetObjectClass(env, set);

  fid = (*env)->GetFieldID(env, class, "set", "I");

  eventSet = (*env)->GetIntField(env, set, fid);

  v_arr = (*env)->GetLongArrayElements(env, values, 0);

  ret = (*papi_write)(eventSet, v_arr);

  (*env)->ReleaseLongArrayElements(env, values, v_arr, 0);

  return ret;
}
