public class PAPI_preset_info {
  public String event_name;
  public int event_code;
  public String event_descr;
  public int avail;
  public String event_note;
  public int flags;

  public PAPI_preset_info() {
    // do nothing
  }

  public PAPI_preset_info(String n, int c, String d, int a, String note, int f)
  {
    event_name = n;
    event_code = c;
    event_descr = d;
    avail = a;
    event_note = note;
    flags = f;
  }
}
