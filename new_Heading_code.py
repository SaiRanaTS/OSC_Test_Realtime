import asyncio
import websockets
import time
import json
import math
import traceback
import itertools
import numpy as np
#from buffer import Buffer
# from osc_demo import OwnShip
# from osc_demo import CtrlCmd

def view_actor_data(actor, port_type, port_name):
    pass
def find_ori_port_value(actor, port_type, port_name_ls):
    port_list = actor[port_type]
    num_port = len(port_list)
    num_port_name = len(port_name_ls)
    orientation = []
    for i in range(num_port_name):
        for j in range(num_port):
            port_name = port_list[j]['port']['name']
            if port_name == port_name_ls[i]:
                if port_name == "ORIENTATION".upper():
                    value_ls = port_list[j]['value']['valueObjects']
                    for v in value_ls:
                        orientation.append(v['value'])
                else:
                    break
    return math.degrees(orientation[2])

def find_gps_port_value(actor, port_type, port_name_ls):
    port_list = actor[port_type]
    num_port = len(port_list)
    num_port_name = len(port_name_ls)
    speed = []
    gps_info = []
    result = [[], []]
    for i in range(num_port_name):
        for j in range(num_port):
            port_name = port_list[j]['port']['name']
            if port_name == port_name_ls[i]:
                if port_name == "WORLD_VELOCITY".upper():
                    value_ls = port_list[j]['value']['valueObjects']
                    for v in value_ls:
                        speed.append(v['value'])
                    result[1] = speed
                elif port_name == "ORIENTATION".upper():
                    break
                else:
                    gps_info.append(port_list[j]['value']['value'])
                    result[0] = gps_info
                    break
    return result

def find_actuator_port_value(actor, port_type, port_name_ls):
    port_list = actor[port_type]
    num_port = len(port_list)
    num_port_name = len(port_name_ls)
    result = []
    for i in range(num_port_name): # 5, 2
        for j in range(num_port): # 34, 34
            port_name = port_list[j]['port']['name']
            if port_name == port_name_ls[i]:
                if port_name == "COMMANDED_ANGLE".upper():
                    angle = port_list[j]['value']['value']
                    result.append(angle)
                elif port_name == "COMMANDED_RPM".upper():
                    velocity = port_list[j]['value']['value']
                    result.append(velocity)
    return result

def ls_to_dic(receivedata, port_gps_info):
    num_port = len(receivedata)
    num_port_name = len(port_gps_info)
    result = dict()
    for i in range(0, num_port, num_port_name):
        for key, val in zip(port_gps_info,receivedata[i:i+num_port_name]):
            if key not in result:
                result[key] = []
            result[key].append(val)
    return result

async def start():
    uri = "ws://192.168.114.18:8887"
    actor_info = {
        'clazz' : '',
        'name' : '',
        'uuid' : None,
        'parent_uuid' : None
    }

    port_gps_info = {
        'clazzname': '',
        'LONGITUDE': None,
        'LATITUDE': None,
        'EASTING': None,
        'NORTHING': None,
        # 'BEARING': None,
        'WORLD_VELOCITY': [],
        'ORIENTATION':[]
    }
    # port_ori_info = {
    #     'clazzname': '',
    #     'ORIENTATION': []
    # }

    port_actuator_info = {
        'clazzname': '',
        # 'ANGLE': [], # starboard rudder, port rudder
        # 'ACTUAL_RPM': [] # starboard rpm, port rpm
        'COMMANDED_ANGLE': [], # starboard rudder, port rudder
        'COMMANDED_RPM': [] # starboard rpm, port rpm
    }

    port_gps_name_ls, port_actuator_name_ls = [], []
    for name in port_gps_info:
        port_gps_name_ls.append(name)
    port_gps_name_ls.pop(0) # port's name
    for name in port_actuator_info:
        port_actuator_name_ls.append(name)
    port_actuator_name_ls.pop(0)
    # for name in port_ori_info:
    #     port_ori_info_name_ls.append(name)
    # port_ori_info_name_ls.pop(0)

    gps_gunnerus_ori = actor_info.copy()
    gps_gunnerus_ori['clazz'] = 'TypedSixDOFActor'
    gps_gunnerus_ori['name'] = 'Gunnerus'

    gps_gunnerus = actor_info.copy()
    gps_gunnerus['clazz'] = 'GPSController'
    gps_gunnerus['name'] = 'GPS1'

    gps_target_ship_1 = actor_info.copy()
    gps_target_ship_1['clazz'] = 'GPSController'
    gps_target_ship_1['name'] = 'Target Ship 1'

    gps_target_ship_2 = actor_info.copy()
    gps_target_ship_2['clazz'] = 'GPSController'
    gps_target_ship_2['name'] = 'Target Ship 2'

    gps_target_ship_3 = actor_info.copy()
    gps_target_ship_3['clazz'] = 'GPSController'
    gps_target_ship_3['name'] = 'Target Ship 3'

    gps_target_ship_4 = actor_info.copy()
    gps_target_ship_4['clazz'] = 'GPSController'
    gps_target_ship_4['name'] = 'Target Ship 4'

    gps_target_ship_5 = actor_info.copy()
    gps_target_ship_5['clazz'] = 'GPSController'
    gps_target_ship_5['name'] = 'Target Ship 5'

    gunnerus_thruster_port = actor_info.copy()
    gunnerus_thruster_port['clazz'] = 'ThrusterActor'
    gunnerus_thruster_port['name'] = 'Port'

    gunnerus_thruster_starboard = actor_info.copy()
    gunnerus_thruster_starboard['clazz'] = 'ThrusterActor'
    gunnerus_thruster_starboard['name'] = 'Starboard'

    actor_info_list = [gps_gunnerus, gps_gunnerus_ori, gps_target_ship_1, gps_target_ship_2, gps_target_ship_3, gps_target_ship_4, gps_target_ship_5, gunnerus_thruster_port, gunnerus_thruster_starboard]
    port_value = []
    async with websockets.connect(uri, ping_timeout=None) as websocket:
        while True:
            if not websocket.open:
                print('reconnecting')
                websocket = await websockets.connect(uri)
            else:
                resp = await websocket.recv()
                #print(resp)
                actuator_set_json = None
                actuators_set_json = []
                actuator_get_json = []
                try:
                    data_dic = json.loads(resp[resp.index('{'):])
                    for i in range(8):
                        actor_info = actor_info_list[i]
                        actor = await evaluate_actor(data_dic, actor_info['clazz'], actor_info['name']) # dic
                        if actor != None:
                            if actor['name'] == 'Starboard' or actor['name'] == 'Port':
                                actuator_get_json.append(actor)
                                # actuator_set_json = set_actuator_json(actor, 'input', 90)
                                # actuators_set_json.append(actuator_set_json)
                                # actuator_port_value = find_actuator_port_value(actor, 'output', port_actuator_name_ls)
                                # port_value.append(actuator_port_value)
                                # actuator_port_value = find_actuator_port_value(actor, 'input', port_actuator_name_ls)
                                # print(actor)
                                # port_value.append(actuator_port_value)
                                # print(port_value)
                            # elif actor['clazz'].__contains__('TypedSixDOFActor'):
                            elif actor['clazz'].__contains__('TypedSixDOFActor'):
                                gps_gunnerus_orientation = find_ori_port_value(actor, 'output', port_gps_name_ls)
                                print(gps_gunnerus_orientation)
                            else:
                                gps_port_value_rest = find_gps_port_value(actor, 'output', port_gps_name_ls)
                                #ls_to_dic(gps_port_value, port_gps_name_ls)
                                port_value.append(gps_port_value_rest)
                                #print(port_value[0])
                except:
                    traceback.print_exc()
                # save the actuation factor
                await save_json_file(actuator_get_json)

                # with open('my_file.json') as json_file:
                #     data= json.load(json_file)
                #     print(data)
                #     await websocket.send(json.dumps(data))

                # if actuators_set_json != None:
                #     for i in range(len(actuators_set_json)):
                #         #print(actuators_set_json[i])
                #         await websocket.send(json.dumps(actuators_set_json[i]))


async def evaluate_actor(data_dic, clazz, name):
    x = False if data_dic['clazz'].find(clazz) == -1 else True
    y = (data_dic['name'] == name)
    if x and y:
        return data_dic

def set_actuator_json(actor, port_type, deg):
    port_list = actor[port_type]
    num_port = len(port_list)
    for i in range(num_port):
        port_name = port_list[i]['port']['name']
        if port_name == "COMMANDED_ANGLE".upper():
            port_list[i]['value']['value'] = deg*3.14/180
            # print(port_list[i]['value']['value'])
        elif port_name == "COMMANDED_RPM".upper():
            port_list[i]['value']['value'] = 100
            # print(port_list[i]['value']['value'])
    return actor

def save_ls_file(receivedata): # list
    with open('my_file.txt', 'a') as f:
        for item in receivedata:
            f.write("%s\n" % item)

async def save_json_file(receivedata): # di
    with open('myfile.json', 'a') as f:
        for item in receivedata:
            json.dump(item, f)
    # f.close()
def trans_json(actuator_get_json):
    return json.loads(actuator_get_json[actuator_get_json.index('['):])

if __name__=='__main__':
    #rospy.init_node("simulator_drl")
    asyncio.get_event_loop().run_until_complete(start())
    asyncio.get_event_loop().run_forever()
