public class PAPI_exe_info {
  public String fullname;
  public PAPI_address_map address_info;

  public PAPI_exe_info() {
    // do nothing
  }

  public PAPI_exe_info(String f, String n, long ts, long te, long ds,
     long de, long bs, long be)
  {
    fullname = f;
    address_info = new PAPI_address_map(n, ts, te, ds, de, bs, be);
  }

  public PAPI_exe_info(String f)
  {
    fullname = f;
  }

  public PAPI_exe_info(String f, PAPI_address_map a)
  {
    fullname = f;
    address_info = a;
  }
}
