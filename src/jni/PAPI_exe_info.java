public class PAPI_exe_info {
  public String fullname;
  public String name;
  public long text_start;
  public long text_end;
  public long data_start;
  public long data_end;
  public long bss_start;
  public long bss_end;
  public String lib_preload_env; 

  public PAPI_exe_info(String f, String n, long ts, long te, long ds,
     long de, long bs, long be, String lib)
  {
    fullname = f;
    name = n;
    text_start = ts;
    text_end = te;
    data_start = ds;
    data_end = de;
    bss_start = bs;
    bss_end = be;
    lib_preload_env = lib;
  }
}
