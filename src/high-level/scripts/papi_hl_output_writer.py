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

def create_json_object(source):
  json_object = {}
  json_object["ranks"] = []

  #get measurement files
  file_list = os.listdir(source)
  for item in file_list:
    json_rank = {}

    #determine mpi rank based on file name (rank_#)
    rank = item.split('_', 1)[1]
    rank = rank.rsplit('.', 1)[0]
    json_rank["id"] = rank
    json_rank["threads"] = []
    
    #open meaurement file
    file_name = str(source) + "/rank_" + str(rank) 
    try:
      rank_file = open(file_name, "r")
      lines = rank_file.readlines() #[2:]
    except IOError as ioe:
      print("Cannot open file {} ({})".format(file_name, repr(ioe)))
      return
    
    line_counter = 0

    #read lines from file
    for line in lines:
      if line_counter == 0:
        #determine cpu frequency
        global cpu_freq
        cpu_freq = int(line.split(':', 1)[1]) * 1000000
        #print cpu_freq
        line_counter = line_counter + 1
        continue
      if line_counter == 1:
        #skip second line
        line_counter = line_counter + 1
        continue

      thread_id = line.split(',', 1)[0]
      json_thread = {}
      json_thread["id"] = int(thread_id)
      json_thread["regions"] = []

      #remove thread_id from line
      line = line.split(',', 1)[1]

      #load json per thread
      regions = json.loads(line, object_pairs_hook=OrderedDict)
      #print json.dumps(regions)
      for region_key,region_value in regions.items():
        region_dict = OrderedDict()
        region_dict["name"] = region_key
        #print region_key
        event_dict = OrderedDict()
        #print region_value
        for event_key,event_value in region_value.items():
          #print event_key
          event_dict.update({event_key:event_value})

        region_dict["events"] = event_dict
        json_thread["regions"].append(region_dict)

      json_rank["threads"].append(json_thread)

    rank_file.close()
    json_object["ranks"].append(json_rank)

  # print json.dumps(json_object,indent=2, sort_keys=False,
  #                   separators=(',', ': '), ensure_ascii=False)
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
        metric_value = float(value['Total'])
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
      for event_key,event_value in known_events.iteritems():
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
        sum_cnt.add_region(ranks['id'], regions['name'], regions['events'])
  return sum_cnt.get_json()


def get_ipc_dict(inst, cyc):
  ipc_dict = OrderedDict()
  for (inst_key,inst_value), (cyc_key,cyc_value) in zip(inst.items(), cyc.items()):
    #print str(inst_key) + "," + str(inst_value)
    #print str(cyc_key) + "," + str(cyc_value)
    ipc = float(int(inst_value) / int(cyc_value))
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
  if 'REGION_COUNT' in events:
    format_events['Region count'] = int(events['REGION_COUNT'])
    del events['REGION_COUNT']

  #Real Time
  if 'CYCLES' in events:
    if isinstance(events['CYCLES'],dict):
      for read_key,read_value in events['CYCLES'].items():
        rt_dict[read_key] = float(read_value) / int(cpu_freq)
      format_events['Real time in s'] = format_read_events(events['CYCLES'],'Runtime')
    else:
      rt = float(events['CYCLES']) / int(cpu_freq)
      format_events['Real time in s'] = convert_value(events['CYCLES'], 'Runtime')
    del events['CYCLES']

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
      ipc = float(int(events['PAPI_TOT_INS']) / int(events['PAPI_TOT_CYC']))
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
  
  #read the rest
  for event_key,event_value in events.iteritems():
    if isinstance(event_value,dict):
      format_events[event_key] = format_read_events(event_value)
    else:
      format_events[event_key] = convert_value(event_value)

  return format_events


def format_json_object(json):
  json_object = {}
  json_object['ranks'] = []

  for ranks in json['ranks']:
    #print ranks['id']
    json_rank = {}
    json_rank['id'] = ranks['id']
    json_rank['threads'] = []
    for threads in ranks['threads']:
      #print threads['id']
      json_thread = {}
      json_thread['id'] = threads['id']
      json_thread['regions'] = []
      for regions in threads['regions']:
        #print regions['name']
        region = {}
        region['name'] = regions['name']
        region['events'] = {}
        events = {}
        for event_key,event_value in regions['events'].iteritems():
          events[event_key] = event_value
        
        formated_events = format_events(events)

        region['events'] = formated_events
        json_thread['regions'].append(region)
      json_rank['threads'].append(json_thread)
    json_object["ranks"].append(json_rank)
    
  return json_object

def write_json_file(data, file_name):
  with io.open(file_name, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(data,
                      indent=4, sort_keys=False,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))
    print str_


def main(source, format, type):
  if (format == "json"):
    json = create_json_object(source)
    formated_json = format_json_object(json)

    if type == 'detail':
      write_json_file(formated_json, 'papi.json')

    #summarize data over threads and ranks
    if type == 'accumulated':
      sum_json = sum_json_object(formated_json)
      write_json_file(sum_json, 'papi_sum.json')
  else:
    print("Format not supported!")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, required=False, default="papi",
                      help='Measurement directory of raw data.')
  parser.add_argument('--format', type=str, required=False, default='json', 
                      help='Output format, e.g. json.')
  parser.add_argument('--type', type=str, required=False, default='detail', 
                      help='Output type: detail or accumulated.')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(format=args.format,
       source=args.source,
       type=args.type)