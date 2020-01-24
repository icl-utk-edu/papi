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

class Sum_Counters(object):
  regions = OrderedDict()
  regions_last_rank_id = {}

  def add_region(self, rank_id, region, events=OrderedDict()):

    #clean events from read values caused by PAPI_hl_read
    cleaned_events = OrderedDict()
    for key,value in events.items():
      metric_value = value
      if isinstance(value, dict):
        metric_value = float(value['total'])
      cleaned_events[key] = metric_value

    if region not in self.regions:
      #new region
      new_region_events = cleaned_events.copy()
      new_region_events['Number of ranks'] = 1
      new_region_events['Number of threads'] = 1
      new_region_events['Number of processes'] = 1
      self.regions[region] = new_region_events.copy()
      self.regions_last_rank_id[region] = rank_id
    else:
      #add counter values to existing region
      known_events = self.regions[region].copy()
      new_events = cleaned_events.copy()

      #increase number of ranks when rank_id has changed
      if self.regions_last_rank_id[region] == rank_id:
        new_events['Number of ranks'] = 0
      else:
        self.regions_last_rank_id[region] = rank_id
        new_events['Number of ranks'] = 1

      #always increase number of threads
      new_events['Number of threads'] = 1
      new_events['Number of processes'] = 1

      #add values
      for event_key,event_value in known_events.items():
        if 'Number of' in event_key or 'count' in event_key:
          known_events[event_key] = event_value + new_events[event_key]
        else:
          known_events[event_key] = float(format(event_value + new_events[event_key], '.2f'))
      self.regions[region] = known_events.copy()

  def get_json(self):
    #calculate correct thread number (number of processes / number of ranks)
    for name in self.regions:
      events = self.regions[name]
      events['Number of threads'] = int(events['Number of processes'] / events['Number of ranks'])
    return self.regions


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


def format_read_events(events, event_type = 'Other'):
  format_read_dict = OrderedDict()
  for read_key,read_value in events.items():
    format_read_dict[read_key] = convert_value(read_value, event_type)
  return format_read_dict


def format_events(events):
  #keep order as declared
  format_events = OrderedDict()
  #remember runtime for other metrics like MFLOPS
  rt = 1.0
  rt_dict = OrderedDict()

  #Region Count
  if 'region_count' in events:
    format_events['Region count'] = int(events['region_count'])
    del events['region_count']

  #Real Time
  if 'cycles' in events:
    if isinstance(events['cycles'],dict):
      for read_key,read_value in events['cycles'].items():
        rt_dict[read_key] = float(read_value) / int(cpu_freq)
      format_events['Real time in s'] = format_read_events(events['cycles'],'Runtime')
    else:
      rt = float(events['cycles']) / int(cpu_freq)
      format_events['Real time in s'] = convert_value(events['cycles'], 'Runtime')
    del events['cycles']

  #CPU Time
  if 'perf::TASK-CLOCK' in events:
    if isinstance(events['perf::TASK-CLOCK'],dict):
      format_events['CPU time in s'] = format_read_events(events['perf::TASK-CLOCK'],'CPUtime')
    else:
      format_events['CPU time in s'] = convert_value(events['perf::TASK-CLOCK'], 'CPUtime')
    del events['perf::TASK-CLOCK']

  #PAPI_TOT_INS and PAPI_TOT_CYC to calculate IPC
  if 'PAPI_TOT_INS' and 'PAPI_TOT_CYC' in events:
    if isinstance(events['PAPI_TOT_INS'],dict) and isinstance(events['PAPI_TOT_CYC'],dict):
      ipc_dict = get_ipc_dict(events['PAPI_TOT_INS'], events['PAPI_TOT_CYC'])
      format_events['IPC'] = ipc_dict
    else:
      try:
        ipc = float(int(events['PAPI_TOT_INS']) / int(events['PAPI_TOT_CYC']))
      except:
        ipc = 0
      format_events['IPC'] = float(format(ipc, '.2f'))

    del events['PAPI_TOT_INS']
    del events['PAPI_TOT_CYC']
  
  #FLIPS
  if 'PAPI_FP_INS' in events:
    if isinstance(events['PAPI_FP_INS'],dict):
      mflips_dict = get_ops_dict(events['PAPI_FP_INS'], rt_dict)
      format_events['MFLIPS/s'] = mflips_dict
    else:
      mflips = float(events['PAPI_FP_INS']) / 1000000 / rt
      mflips = float(format(mflips, '.2f'))
      format_events['MFLIPS/s'] = mflips
    del events['PAPI_FP_INS']

  #SP vector instructions per second
  if 'PAPI_VEC_SP' in events:
    if isinstance(events['PAPI_VEC_SP'],dict):
      mvecins_dict = get_ops_dict(events['PAPI_VEC_SP'], rt_dict)
      format_events['Single precision vector/SIMD instructions rate in M/s'] = mvecins_dict
    else:
      mvecins = float(events['PAPI_VEC_SP']) / 1000000 / rt
      mvecins = float(format(mvecins, '.2f'))
      format_events['Single precision vector/SIMD instructions rate in M/s'] = mvecins
    del events['PAPI_VEC_SP']

  #DP vector instructions per second
  if 'PAPI_VEC_DP' in events:
    if isinstance(events['PAPI_VEC_DP'],dict):
      mvecins_dict = get_ops_dict(events['PAPI_VEC_DP'], rt_dict)
      format_events['Double precision vector/SIMD instructions rate in M/s'] = mvecins_dict
    else:
      mvecins = float(events['PAPI_VEC_DP']) / 1000000 / rt
      mvecins = float(format(mvecins, '.2f'))
      format_events['Double precision vector/SIMD instructions rate in M/s'] = mvecins
    del events['PAPI_VEC_DP']
  
  #FLOPS
  if 'PAPI_FP_OPS' in events:
    if isinstance(events['PAPI_FP_OPS'],dict):
      mflops_dict = get_ops_dict(events['PAPI_FP_OPS'], rt_dict)
      format_events['MFLOPS/s'] = mflops_dict
    else:
      mflops = float(events['PAPI_FP_OPS']) / 1000000 / rt
      mflops = float(format(mflops, '.2f'))
      format_events['MFLOPS/s'] = mflops
    del events['PAPI_FP_OPS']
  
  #SP FLOPS
  if 'PAPI_SP_OPS' in events:
    if isinstance(events['PAPI_SP_OPS'],dict):
      mflops_dict = get_ops_dict(events['PAPI_SP_OPS'], rt_dict)
      format_events['Single precision MFLOPS/s'] = mflops_dict
    else:
      mflops = float(events['PAPI_SP_OPS']) / 1000000 / rt
      mflops = float(format(mflops, '.2f'))
      format_events['Single precision MFLOPS/s'] = mflops
    del events['PAPI_SP_OPS']

  #DP FLOPS
  if 'PAPI_DP_OPS' in events:
    if isinstance(events['PAPI_DP_OPS'],dict):
      mflops_dict = get_ops_dict(events['PAPI_DP_OPS'], rt_dict)
      format_events['Double precision MFLOPS/s'] = mflops_dict
    else:
      mflops = float(events['PAPI_DP_OPS']) / 1000000 / rt
      mflops = float(format(mflops, '.2f'))
      format_events['Double precision MFLOPS/s'] = mflops
    del events['PAPI_DP_OPS']

  #read the rest
  for event_key,event_value in events.items():
    if isinstance(event_value,dict):
      format_events[event_key] = format_read_events(event_value)
    else:
      format_events[event_key] = convert_value(event_value)

  return format_events


def format_json_object(json):
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
          json_region[region_key] = format_events(region_value)

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


def main(source, format, type):
  if (format == "json"):
    json = merge_json_files(source)
    formated_json = format_json_object(json)

    if type == 'detail':
      write_json_file(formated_json, 'papi.json')

    #summarize data over threads and ranks
    if type == 'accumulate':
      sum_json = sum_json_object(formated_json)
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
                      help='Output type: detail or accumulate.')

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
  if output_type != "detail" and output_type != "accumulate":
    print("Output type '{}' is not supported!\n".format(output_type))
    parser.print_help()
    parser.exit()
  

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(format=args.format,
       source=args.source,
       type=args.type)