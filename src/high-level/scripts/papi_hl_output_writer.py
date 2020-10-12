#!/usr/bin/python
from __future__ import division
from collections import OrderedDict

import argparse
import os
import json
# Make it work for Python 2+3 and with Unicode
import io
try:
  to_unicode = unicode
except NameError:
  to_unicode = str

cpu_freq = 0
event_definitions = {}
process_num = {}

event_rate_names = OrderedDict([
      ('PAPI_FP_INS','MFLIPS/s'),
      ('PAPI_VEC_SP','Single precision vector/SIMD instructions rate in M/s'),
      ('PAPI_VEC_DP','Double precision vector/SIMD instructions rate in M/s'),
      ('PAPI_FP_OPS','MFLOPS/s'),
      ('PAPI_SP_OPS','Single precision MFLOPS/s'),
      ('PAPI_DP_OPS','Double precision MFLOPS/s')
    ])

def merge_json_files(source):
  json_object = {}
  json_object["ranks"] = []
  events_stored = False

  #get measurement files
  file_list = os.listdir(source)
  for item in file_list:
    json_rank = {}

    #determine mpi rank based on file name (rank_#)
    rank = item.split('_', 1)[1]
    rank = rank.rsplit('.', 1)[0]
    #print("rank: {}".format(rank))

    json_rank["id"] = rank

    #open measurement file
    file_name = str(source) + "/rank_" + str(rank) 

    try:
      with open(file_name) as json_file:
        #keep order of all objects
        data = json.load(json_file, object_pairs_hook=OrderedDict)
    except IOError as ioe:
      print("Cannot open file {} ({})".format(file_name, repr(ioe)))
      return

    #store global data
    if events_stored == False:
      #determine cpu frequency
      global cpu_freq
      cpu_freq = int(data['cpu in mhz']) * 1000000
      global event_definitions
      event_definitions = data['event_definitions']
      events_stored = True

    #get all threads
    json_rank["threads"] = data["threads"]

    #append current rank to json file
    json_object["ranks"].append(json_rank)

  # print json.dumps(json_object,indent=2, sort_keys=False,
  #                  separators=(',', ': '), ensure_ascii=False)
  return json_object

class Sum_Counter(object):
  def __init__(self):
    self.min = None
    self.all_values = []
    self.max = 0

  def add_event(self, value):
    if isinstance(value, dict):
      if self.min > long(value['min']) or self.min is None:
        self.min = long(value['min'])
      self.all_values.append(long(value['avg']))
      if self.max < long(value['max']):
        self.max = long(value['max'])
    else:
      val = long(value)
      if self.min > val or self.min is None:
        self.min = val
      self.all_values.append(val)
      if self.max < val:
        self.max = val

  def get_min(self):
    return self.min

  def get_median(self):
    n = len(self.all_values)
    s = sorted(self.all_values)
    return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None

  def get_sum(self):
    sum = 0
    for value in self.all_values:
      sum += value
    return sum

  def get_max(self):
    return self.max


class Sum_Counters(object):

  def __init__(self):
    self.regions = OrderedDict()
    self.regions_last_rank_id = {}
    self.regions_rank_num = {}
    self.regions_thread_num = {}

    self.clean_regions = OrderedDict()
    self.sum_counters = OrderedDict()

  def add_region(self, rank_id, region, events=OrderedDict()):

    #remove all read values caused by PAPI_hl_read
    cleaned_events = OrderedDict()
    for key,value in events.items():
      metric_value = value
      if isinstance(value, dict):
        if "total" in value:
          metric_value = float(value['total'])
        elif "min" in value and "avg" in value and "max" in value:
          metric_value = {"min":value['min'], "avg":value['avg'], "max":value['max']}
        else:
          metric_value = value
      cleaned_events[key] = metric_value
      #print("add_region ", rank_id, region, key, metric_value)

    #create new Sum_Counter object for each new region
    if region not in self.regions:
      self.regions[region] = {}
      self.regions_last_rank_id[region] = rank_id
      self.regions_rank_num[region] = 1
      self.regions_thread_num[region] = 1
      self.sum_counters[region] = OrderedDict()
      for key,value in cleaned_events.items():
        self.sum_counters[region][key] = Sum_Counter()
        self.sum_counters[region][key].add_event(value)
    else:
      #increase number of ranks when rank_id has changed
      if self.regions_last_rank_id[region] != rank_id:
        self.regions_last_rank_id[region] = rank_id
        self.regions_rank_num[region] += 1

      #always increase number of threads
      self.regions_thread_num[region] += 1

      for key,value in cleaned_events.items():
        self.sum_counters[region][key].add_event(value)

    self.regions[region]['rank_num'] = self.regions_rank_num[region]
    self.regions[region]['thread_num'] = self.regions_thread_num[region]


  def get_json(self):
    sum_json = OrderedDict()
    for name in self.regions:
      events = OrderedDict()
      for key,value in self.sum_counters.items():
        if key == name:
          region_count = 1
          for event_key,event_value in value.items():
            if event_key == 'region_count':
              events[event_key] = int(event_value.get_sum())
              region_count = events[event_key]
            else:
              global event_definitions
              if self.regions[name]['rank_num'] > 1 or self.regions[name]['thread_num'] > 1:
                events[event_key] = OrderedDict()
                if event_key == 'cycles':
                  events[event_key]['total'] = event_value.get_sum()
                else:
                  if event_definitions[event_key] == 'delta':
                    events[event_key]['total'] = event_value.get_sum()
                events[event_key]['min'] = event_value.get_min()
                events[event_key]['median'] = event_value.get_median()
                events[event_key]['max'] = event_value.get_max()
              else:
                #sequential code
                if event_key == 'cycles':
                  events[event_key] = event_value.get_min()
                else:
                  if event_definitions[event_key] == 'instant' and region_count > 1:
                    events[event_key] = OrderedDict()
                    events[event_key]['min'] = event_value.get_min()
                    events[event_key]['median'] = event_value.get_median()
                    events[event_key]['max'] = event_value.get_max()
                  else:
                    events[event_key] = event_value.get_min()
          break

      #add number of ranks and threads in case of a parallel code
      if self.regions[name]['rank_num'] > 1 or self.regions[name]['thread_num'] > 1:
        events['Number of ranks'] = self.regions[name]['rank_num']
        events['Number of threads per rank'] = int(self.regions[name]['thread_num'] / self.regions[name]['rank_num'])
      sum_json[name] = events

      global process_num
      process_num[name] = self.regions[name]['rank_num'] * self.regions[name]['thread_num']
    return sum_json

def derive_sum_json_object(json):
  json_object = OrderedDict()

  for region_key,region_value in json.items():
    #print("region: ", region_key)
    derive_events = OrderedDict()
    events = region_value.copy()
    global process_num
    proc_num = process_num[region_key]

    #remember runtime for other metrics like MFLOPS
    rt = {}

    #Region Count
    if 'region_count' in events:
      derive_events['Region count'] = int(events['region_count'])
      del events['region_count']

    #Real Time
    if 'cycles' in events:
      event_name = 'Real time in s'
      if proc_num > 1:
        for metric in ['total', 'min', 'median', 'max']:
          rt[metric] = convert_value(events['cycles'][metric], 'Runtime')
        derive_events[event_name] = rt['max']
      else:
        rt['total'] = convert_value(events['cycles'], 'Runtime')
        derive_events[event_name] = rt['total']
      del events['cycles']

    #CPU Time
    if 'perf::TASK-CLOCK' in events:
      event_name = 'CPU time in s'
      if proc_num > 1:
        derive_events['CPU time in s'] = convert_value(events['perf::TASK-CLOCK']['total'], 'CPUtime')
      else:
        derive_events['CPU time in s'] = convert_value(events['perf::TASK-CLOCK'], 'CPUtime')
      del events['perf::TASK-CLOCK']

    #PAPI_TOT_INS and PAPI_TOT_CYC to calculate IPC
    if 'PAPI_TOT_INS' in events and 'PAPI_TOT_CYC' in events:
      event_name = 'IPC'
      metric = 'total'
      try:
        if proc_num > 1: 
          ipc = float(format(float(int(events['PAPI_TOT_INS'][metric]) / int(events['PAPI_TOT_CYC'][metric])), '.2f'))
        else:
          ipc = float(format(float(int(events['PAPI_TOT_INS']) / int(events['PAPI_TOT_CYC'])), '.2f'))
      except:
        ipc = 'n/a'
      derive_events[event_name] = ipc

      del events['PAPI_TOT_INS']
      del events['PAPI_TOT_CYC']
    
    #Rates
    global event_rate_names
    for rate_event in event_rate_names:
      if rate_event in events:
        event_name = event_rate_names[rate_event]
        metric = 'total'
        try:
          if proc_num > 1:
            rate = float(format(float(events[rate_event][metric]) / 1000000 / rt[metric], '.2f'))
          else:
            rate = float(format(float(events[rate_event]) / 1000000 / rt[metric], '.2f'))
        except:
          rate = 'n/a'
        derive_events[event_name] = rate

        del events[rate_event]


    #read the rest
    for event_key,event_value in events.items():
      derive_events[event_key] = OrderedDict()
      derive_events[event_key] = event_value

    json_object[region_key] = derive_events.copy()

  return json_object

def sum_json_object(json, derived = False):
  sum_cnt = Sum_Counters()
  for ranks in json['ranks']:
    for threads in ranks['threads']:
      for regions in threads['regions']:
        for region_key,region_value in regions.items():
          name = region_key
          events = region_value
          sum_cnt.add_region(ranks['id'], name, events)

  if derived == True:
    return derive_sum_json_object(sum_cnt.get_json())
  else:
    return sum_cnt.get_json()


def get_ipc_dict(inst, cyc):
  ipc_dict = OrderedDict()
  for (inst_key,inst_value), (cyc_key,cyc_value) in zip(inst.items(), cyc.items()):
    #print str(inst_key) + "," + str(inst_value)
    #print str(cyc_key) + "," + str(cyc_value)
    try:
      ipc = float(int(inst_value) / int(cyc_value))
    except:
      ipc = 0
    ipc_dict[inst_key] = float(format(ipc, '.2f'))
  return ipc_dict


def get_ops_dict(ops, rt):
  ops_dict = OrderedDict()
  for (ops_key,ops_value), (rt_key,rt_value) in zip(ops.items(), rt.items()):
    #print str(ops_key) + "," + str(ops_value)
    #print str(rt_key) + "," + str(rt_value)
    ops = float(ops_value) / 1000000 / rt_value
    ops_dict[ops_key] = float(format(ops, '.2f'))
  return ops_dict


def convert_value(value, event_type = 'Other'):
  global cpu_freq
  if event_type == 'Other':
    result = float(value)
    result = float(format(result, '.2f'))
  elif event_type == 'Runtime':
    try:
      result = float(value) / int(cpu_freq)
    except:
      result = 1.0
    result = float(format(result, '.2f'))
  elif event_type == 'CPUtime':
    result = float(value) / 1000000000
    result = float(format(result, '.2f'))

  return result


def derive_read_events(events, event_type = 'Other'):
  format_read_dict = OrderedDict()
  for read_key,read_value in events.items():
    format_read_dict[read_key] = convert_value(read_value, event_type)
  return format_read_dict


def derive_events(events):
  global cpu_freq
  #keep order as declared
  derive_events = OrderedDict()
  #remember runtime for other metrics like MFLOPS
  rt = 1.0
  rt_dict = OrderedDict()

  #Region Count
  if 'region_count' in events:
    derive_events['Region count'] = int(events['region_count'])
    del events['region_count']

  #Real Time
  if 'cycles' in events:
    if isinstance(events['cycles'],dict):
      for read_key,read_value in events['cycles'].items():
        rt_dict[read_key] = float(read_value) / int(cpu_freq)
      derive_events['Real time in s'] = derive_read_events(events['cycles'],'Runtime')
    else:
      rt = float(events['cycles']) / int(cpu_freq)
      derive_events['Real time in s'] = convert_value(events['cycles'], 'Runtime')
    del events['cycles']

  #CPU Time
  if 'perf::TASK-CLOCK' in events:
    if isinstance(events['perf::TASK-CLOCK'],dict):
      derive_events['CPU time in s'] = derive_read_events(events['perf::TASK-CLOCK'],'CPUtime')
    else:
      derive_events['CPU time in s'] = convert_value(events['perf::TASK-CLOCK'], 'CPUtime')
    del events['perf::TASK-CLOCK']

  #PAPI_TOT_INS and PAPI_TOT_CYC to calculate IPC
  if 'PAPI_TOT_INS' in events and 'PAPI_TOT_CYC' in events:
    if isinstance(events['PAPI_TOT_INS'],dict) and isinstance(events['PAPI_TOT_CYC'],dict):
      ipc_dict = get_ipc_dict(events['PAPI_TOT_INS'], events['PAPI_TOT_CYC'])
      derive_events['IPC'] = ipc_dict
    else:
      try:
        ipc = float(int(events['PAPI_TOT_INS']) / int(events['PAPI_TOT_CYC']))
      except:
        ipc = 0
      derive_events['IPC'] = float(format(ipc, '.2f'))

    del events['PAPI_TOT_INS']
    del events['PAPI_TOT_CYC']
  
  #Rates
  global event_rate_names
  for rate_event in event_rate_names:
    if rate_event in events:
      event_name = event_rate_names[rate_event]
      if isinstance(events[rate_event],dict):
        rate_dict = get_ops_dict(events[rate_event], rt_dict)
        derive_events[event_name] = rate_dict
      else:
        rate = float(format(float(events[rate_event]) / 1000000 / rt, '.2f'))
        derive_events[event_name] = rate
      del events[rate_event]

  #read the rest
  for event_key,event_value in events.items():
    if isinstance(event_value,dict):
      derive_events[event_key] = derive_read_events(event_value)
    else:
      derive_events[event_key] = convert_value(event_value)

  return derive_events


def derive_json_object(json):
  json_object = {}
  json_object['ranks'] = []

  for rank in json['ranks']:
    # print rank['id']
    # print rank['threads']
    json_rank = {}
    json_rank['id'] = rank['id']
    json_rank['threads'] = []

    for thread in rank['threads']:
      # print thread['id']
      json_thread = {}
      json_thread['id'] = thread['id']
      json_thread['regions'] = []
      for region in thread['regions']:
        json_region = {}
        for region_key,region_value in region.items():
          # print region_key
          # print region_value
          json_region[region_key] = derive_events(region_value)

        json_thread['regions'].append(json_region)
      json_rank['threads'].append(json_thread)
    json_object['ranks'].append(json_rank)

  return json_object

def write_json_file(data, file_name):
  with io.open(file_name, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(data,
                      indent=4, sort_keys=False,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))
    print (str_)


def main(source, format, type, notation):
  if (format == "json"):
    json = merge_json_files(source)

    if type == 'detail':
      if notation == 'derived':
        write_json_file(derive_json_object(json), 'papi.json')
      else:
        write_json_file(json, 'papi.json')

    #summarize data over threads and ranks
    if type == 'summary':
      if notation == 'derived':
        write_json_file(sum_json_object(json, True), 'papi_sum.json')
      else:
        write_json_file(sum_json_object(json), 'papi_sum.json')

  else:
    print("Format not supported!")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, required=False, default="papi_hl_output",
                      help='Measurement directory of raw data.')
  parser.add_argument('--format', type=str, required=False, default='json', 
                      help='Output format, e.g. json.')
  parser.add_argument('--type', type=str, required=False, default='summary', 
                      help='Output type: detail or summary.')
  parser.add_argument('--notation', type=str, required=False, default='derived', 
                      help='Output notation: raw or derived.')

  # check if papi directory exists
  source = str(parser.parse_args().source)
  if os.path.isdir(source) == False:
    print("Measurement directory '{}' does not exist!\n".format(source))
    parser.print_help()
    parser.exit()

  # check format
  output_format = str(parser.parse_args().format)
  if output_format != "json":
    print("Output format '{}' is not supported!\n".format(output_format))
    parser.print_help()
    parser.exit()

  # check type
  output_type = str(parser.parse_args().type)
  if output_type != "detail" and output_type != "summary":
    print("Output type '{}' is not supported!\n".format(output_type))
    parser.print_help()
    parser.exit()

  # check notation
  output_notation = str(parser.parse_args().notation)
  if output_notation != "raw" and output_notation != "derived":
    print("Output notation '{}' is not supported!\n".format(output_notation))
    parser.print_help()
    parser.exit()
  

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(format=args.format,
       source=args.source,
       type=args.type,
       notation=args.notation)