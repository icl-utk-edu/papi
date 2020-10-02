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

def merge_json_files(source):
  json_object = {}
  json_object["ranks"] = []

  #get measurement files
  file_list = os.listdir(source)
  for item in file_list:
    json_rank = {}

    #determine mpi rank based on file name (rank_#)
    rank = item.split('_', 1)[1]
    rank = rank.rsplit('.', 1)[0]
    #print("rank: {}".format(rank))

    json_rank["id"] = rank

    #open meaurement file
    file_name = str(source) + "/rank_" + str(rank) 

    try:
      with open(file_name) as json_file:
        #keep order of all objects
        data = json.load(json_file, object_pairs_hook=OrderedDict)
    except IOError as ioe:
      print("Cannot open file {} ({})".format(file_name, repr(ioe)))
      return

    #determine cpu frequency
    global cpu_freq
    cpu_freq = int(data['cpu in mhz']) * 1000000

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
    self.max = 0.0

  def add_event(self, value):
    if isinstance(value, dict):
      if self.min > value['min'] or self.min is None:
        self.min = value['min']
      self.all_values.append(value['avg'])
      if self.max < value['max']:
        self.max = value['max']
    else:
      if self.min > value or self.min is None:
        self.min = value
      self.all_values.append(value)
      if self.max < value:
        self.max = value

  def get_min(self):
    return self.min

  def get_median(self):
    return self.all_values

  def get_sum(self):
    sum = 0.0
    for value in self.all_values:
      sum += float(value)
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
          for event_key,event_value in value.items():
            if event_key == 'region_count':
              events[event_key] = int(event_value.get_sum())
            else:
              events[event_key] = OrderedDict()
              events[event_key]['min'] = event_value.get_min()
              events[event_key]['median'] = event_value.get_median()
              events[event_key]['max'] = event_value.get_max()
        break
      events['Number of ranks'] = self.regions[name]['rank_num']
      events['Number of threads per rank'] = int(self.regions[name]['thread_num'] / self.regions[name]['rank_num'])
      
      sum_json[name] = events
    return sum_json


def sum_json_object(json):
  sum_cnt = Sum_Counters()
  for ranks in json['ranks']:
    for threads in ranks['threads']:
      for regions in threads['regions']:
        for region_key,region_value in regions.items():
          name = region_key
          events = region_value
          sum_cnt.add_region(ranks['id'], name, events)

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
  if 'PAPI_TOT_INS' and 'PAPI_TOT_CYC' in events:
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
  
  #FLIPS
  if 'PAPI_FP_INS' in events:
    if isinstance(events['PAPI_FP_INS'],dict):
      mflips_dict = get_ops_dict(events['PAPI_FP_INS'], rt_dict)
      derive_events['MFLIPS/s'] = mflips_dict
    else:
      mflips = float(events['PAPI_FP_INS']) / 1000000 / rt
      mflips = float(format(mflips, '.2f'))
      derive_events['MFLIPS/s'] = mflips
    del events['PAPI_FP_INS']

  #SP vector instructions per second
  if 'PAPI_VEC_SP' in events:
    if isinstance(events['PAPI_VEC_SP'],dict):
      mvecins_dict = get_ops_dict(events['PAPI_VEC_SP'], rt_dict)
      derive_events['Single precision vector/SIMD instructions rate in M/s'] = mvecins_dict
    else:
      mvecins = float(events['PAPI_VEC_SP']) / 1000000 / rt
      mvecins = float(format(mvecins, '.2f'))
      derive_events['Single precision vector/SIMD instructions rate in M/s'] = mvecins
    del events['PAPI_VEC_SP']

  #DP vector instructions per second
  if 'PAPI_VEC_DP' in events:
    if isinstance(events['PAPI_VEC_DP'],dict):
      mvecins_dict = get_ops_dict(events['PAPI_VEC_DP'], rt_dict)
      derive_events['Double precision vector/SIMD instructions rate in M/s'] = mvecins_dict
    else:
      mvecins = float(events['PAPI_VEC_DP']) / 1000000 / rt
      mvecins = float(format(mvecins, '.2f'))
      derive_events['Double precision vector/SIMD instructions rate in M/s'] = mvecins
    del events['PAPI_VEC_DP']
  
  #FLOPS
  if 'PAPI_FP_OPS' in events:
    if isinstance(events['PAPI_FP_OPS'],dict):
      mflops_dict = get_ops_dict(events['PAPI_FP_OPS'], rt_dict)
      derive_events['MFLOPS/s'] = mflops_dict
    else:
      mflops = float(events['PAPI_FP_OPS']) / 1000000 / rt
      mflops = float(format(mflops, '.2f'))
      derive_events['MFLOPS/s'] = mflops
    del events['PAPI_FP_OPS']
  
  #SP FLOPS
  if 'PAPI_SP_OPS' in events:
    if isinstance(events['PAPI_SP_OPS'],dict):
      mflops_dict = get_ops_dict(events['PAPI_SP_OPS'], rt_dict)
      derive_events['Single precision MFLOPS/s'] = mflops_dict
    else:
      mflops = float(events['PAPI_SP_OPS']) / 1000000 / rt
      mflops = float(format(mflops, '.2f'))
      derive_events['Single precision MFLOPS/s'] = mflops
    del events['PAPI_SP_OPS']

  #DP FLOPS
  if 'PAPI_DP_OPS' in events:
    if isinstance(events['PAPI_DP_OPS'],dict):
      mflops_dict = get_ops_dict(events['PAPI_DP_OPS'], rt_dict)
      derive_events['Double precision MFLOPS/s'] = mflops_dict
    else:
      mflops = float(events['PAPI_DP_OPS']) / 1000000 / rt
      mflops = float(format(mflops, '.2f'))
      derive_events['Double precision MFLOPS/s'] = mflops
    del events['PAPI_DP_OPS']

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
      sum_json = sum_json_object(json)
      # write_json_file(derive_json_object(sum_json), 'papi_sum.json')
      write_json_file(sum_json, 'papi_sum.json')
  else:
    print("Format not supported!")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, required=False, default="papi_hl_output",
                      help='Measurement directory of raw data.')
  parser.add_argument('--format', type=str, required=False, default='json', 
                      help='Output format, e.g. json.')
  parser.add_argument('--type', type=str, required=False, default='detail', 
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
  

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(format=args.format,
       source=args.source,
       type=args.type,
       notation=args.notation)