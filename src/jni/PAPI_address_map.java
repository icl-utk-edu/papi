public class PAPI_address_map {
  public String name;
  public long text_start;   /* Start address of program text segment */
  public long text_end;      /* End address of program text segment */
  public long data_start;   /* Start address of program data segment */
  public long data_end;      /* End address of program data segment */
  public long bss_start;     /* Start address of program bss segment */
  public long bss_end;        /* End address of program bss segment */

  public PAPI_address_map(String n, long ts, long te, long ds,
     long de, long bs, long be)
  {
    name = n;
    text_start = ts;
    text_end = te;
    data_start = ds;
    data_end = de;
    bss_start = bs;
    bss_end = be;
  }
}
