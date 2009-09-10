public class PAPI_shlib_info {
  public int count;
  public PAPI_address_map map;

  public PAPI_shlib_info() {
    // do nothing
  }

  public PAPI_shlib_info(int c, String n, long ts, long te, long ds,
     long de, long bs, long be)
  {
    count = c;
    map = new PAPI_address_map(n, ts, te, ds, de, bs, be);
  }

  public PAPI_shlib_info(int c)
  {
    count = c;
  }

  public PAPI_shlib_info(int c, PAPI_address_map m)
  {
    count = c;
    map = m;
  }
}
