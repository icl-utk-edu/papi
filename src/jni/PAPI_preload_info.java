public class PAPI_preload_info {
  public String lib_preload_env;   
  public char lib_preload_sep;
  public String lib_dir_env;
  public char lib_dir_sep;

  public PAPI_preload_info() {
    // do nothing
  }

  public PAPI_preload_info(String pe, char ps, String de, char ds)
  {
   lib_preload_env=pe;
   lib_preload_sep=ps;
   lib_dir_env=de;
   lib_dir_sep=ds;
  }
}
