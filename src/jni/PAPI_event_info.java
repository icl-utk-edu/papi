public class PAPI_event_info {
  public int event_code;        /* preset (0x8xxxxxxx) or native (0x4xxxxxxx) event code */
  public int count;              /* number of terms (usually 1) in the code and name fields
                                                - for presets, these terms are native events
                                                - for native events, these terms are register contents */
  public String symbol;                   /* name of the event
                                                - for presets, something like PAPI_TOT_INS
                                                - for native events, something related to the vendor name */
  public String short_descr;            /* a description suitable for use as a label, typically only
                                                implemented for preset events */
  public String long_descr;             /* a longer description of the event
                                                - typically a sentence for presets
                                                - possibly a paragraph from vendor docs for native events */
  public String derived;                 /* name of the derived type
                                                - for presets, usually NOT_DERIVED
                                                - for native events, empty string 
                                                NOTE: a derived description string is available
                                                   in papi_data.c that is currently not exposed to the user */
  public String postfix;                 /* string containing postfix operations; only defined for 
                                                preset events of derived type DERIVED_POSTFIX */
  public int [] code;/* array of values that further describe the event:
                                                - for presets, native event_code values
                                                - for native events, register values for event programming */
  public String [] name;                  /* names of code terms: */
                                           /* - for presets, native event names, as in symbol, above
                                                   NOTE: these may be truncated to fit
                                                - for native events, descriptive strings for each register
                                                   value presented in the code array */
  public String note;                     /* an optional developer note supplied with a preset event
                                                to delineate platform specific anomalies or restrictions
                                                NOTE: could also be implemented for native events. */
  public PAPI_event_info() {
    // do nothing
  }

  public PAPI_event_info(int ecode, int c, String s, String sd, String ld, String d, String p, int [] cd, String [] n, String nt )
  {
    event_code = ecode;
    count = c;
    symbol = s;
    short_descr = sd;
    long_descr = ld;
    derived = d;
    postfix = p;
    code = new int[cd.length];
    for(int i=0; i < cd.length; i++)
        code[i]=cd[i];
    name = new String[n.length];
    for(int i=0; i < n.length; i++)
        name[i]=n[i];
    note=nt;    
  }
}
