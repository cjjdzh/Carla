import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
import cv2
import cv
import carla
import matplotlib.pyplot as plt
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import random
import time
import numpy as np
import random
IM_WIDTH = 1280
IM_HEIGHT = 720
left_fit=None
right_fit=None
#get 10 waypoints in front of the car and store value in look_ahead_list
def update_look_head_list(waypoints,car_location):
    min_dist=float("inf")
    min_way=0
    for way in waypoints:
        dist=np.linalg.norm(np.array([way[0] - car_location[0], way[1] - car_location[1]]))
        if dist<min_dist:
            min_dist=dist
            min_way=way
    idx=waypoints.index(min_way)
    return waypoints[idx:idx+10],idx

# get car location. Format: [x,y,yaw,z]
def get_car_location(vehicle):
    yaw = vehicle.get_transform().rotation.yaw
    if yaw < 0:
        while yaw < 0:
            yaw += 360
    if yaw > 360:
        while yaw > 360:
            yaw -= 360
    return [vehicle.get_location().x,vehicle.get_location().y,yaw,vehicle.get_location().z]

# given a vector, determine whether a point is on the left or right. 1 for right and -1 for left.
def point_to_vector(a,b,p):
    left=-1
    right=1
    x1=b[0]-a[0]
    y1=b[1]-a[1]
    x2=a[0]-p[0]
    y2=a[1]-p[1]
    if x1*y2-x2*y1<0:
        direction=left
    else:
        direction=right
    return direction

# this function is aimed to determine the direction we need to turn
def point_to_vector_dist(a,b,p):
    return abs((b[1]-a[1])*p[0]-(b[0]-a[0])*p[1]+b[0]*a[1]-b[1]*a[0])/np.sqrt((b[1]-a[1])**2+(b[0]-a[0])**2)

# this function is aimed to determine how much we need to set for steer for Kp control
def math_conv(look_ahead_list,car_location):
    look_ahead_list=np.array(look_ahead_list)
    # set a vector pointing from the head waipoint in look_ahead_list to the tail waypoint.
    a=look_ahead_list[0]
    b=look_ahead_list[-1]
    #calculate the average yaw of waypoints in look_ahead_list
    yaw_way=np.average(look_ahead_list[:,2])
    #calculate the distance between vehicle location and the vector we set before, if car is on the left, dist is positive, otherwise is negative
    dist=point_to_vector(a,b,car_location)*point_to_vector_dist(a,b,car_location)
    #print (yaw_way,car_location[2])
    #calculate the yaw difference between car and the vector.
    yaw_diff=np.sin((yaw_way-car_location[2])*np.pi/180)
    output=yaw_diff+dist/5
    if output>1:
        output=1
    if output<-1:
        output=-1
    #print(car_location[2], yaw_way, yaw_diff, output)

    return output


def set_throttle(currentz,lastz):
    #uphill
    if currentz-lastz>0.01:
        throttle=0.6
    #downhill
    elif currentz-lastz<-0.02:
        throttle=0.2
    #normal
    else:
        throttle=0.4
    return throttle


def process_img(image):
    print (type(image))
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    # cv2.imshow("", i3)
    left_fit, right_fit, output1 =cv.find_street_lanes(i3)
    cv2.imshow("",cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
    #plt.imshow(cv2.cvtColor(output1, cv2.COLOR_BGR2RGB))
    #image.save_to_disk("image"+str(random.randint(1,1000)))
    #cv2.imwrite('test.png', i3)
    return None


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)
    spawn_point=random.choice(world.get_map().get_spawn_points())
    a = spawn_point.location
    b = random.choice(world.get_map().get_spawn_points()).location
    #spawn_point = carla.Transform(carla.Location(x=-47.227345, y=-195.189911, z=0.275307), carla.Rotation(pitch=0.000000, yaw=1.439560, roll=0.000000))
    #spawn_point = carla.Transform(carla.Location(x=47.227345, y=-193.189911, z=0.275307), carla.Rotation(pitch=0.000000, yaw=1.439560, roll=0.000000))
    #print (spawn_point.location.x)
    vehicle = world.spawn_actor(bp, spawn_point)
    map = world.get_map()
    vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0))
    #vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
    actor_list.append(vehicle)

    # spectator is used to fix the view in world
    spectator = world.get_spectator()
    camera_bp = world.get_blueprint_library().find('sensor.other.collision')
    camera_transform = carla.Transform(carla.Location(x=-15, z=5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=8.0))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    sensor.listen(lambda data: process_img(data))

    #generate the route based on start point a and end point b
    map = client.get_world().get_map()
    dao = GlobalRoutePlannerDAO(map, 0.5)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    w1 = grp.trace_route(a, b)

    #get all waypoints in the route
    def extractWaypoint(waypoint):
        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        yaw = waypoint.transform.rotation.yaw
        if yaw<0:
            while yaw<0:
                yaw+=360
        if yaw>360:
            while yaw > 360:
                yaw -= 360
        return [x, y, yaw]
    waypoints = []
    for wp in w1:
        #print(extractWaypoint(wp[0]))
        waypoints.append(extractWaypoint(wp[0]))

    #start
    vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0))
    look_ahead_list=waypoints[:10]
    vehicle_location=get_car_location(vehicle)
    look_ahead_list,idx=update_look_head_list(waypoints,vehicle_location)
    last_steer_esti = -1.3245476616753473e-09
    distance=[]
    lastz=0.001

    while 1:
        #get the view of car in world
        spectator.set_transform(camera.get_transform())
        #get traffic light state
        lightsign=vehicle.get_traffic_light()
        if 1!=1:
            #red light set brake to 1
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0,brake=1))
        else:
            if len(look_ahead_list)<5:
                #get to the destination. stop the car
                vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
                break
            #get the 10 waypoints based on current location of car
            look_ahead_list,idx = update_look_head_list(waypoints, vehicle_location)
            #get the car location
            vehicle_location = get_car_location(vehicle)
            #get the z axis of current location
            currentz=vehicle_location[3]
            #get how much need to turn for steer, this value is used for proportional control
            steer_esti=math_conv(look_ahead_list,vehicle_location)
            #get how much need to turn for steer, this value is used for differential control
            K_d_error=steer_esti-last_steer_esti
            #use PD controller
            input_steer=steer_esti+ 1.5*K_d_error
            #set throttle
            input_throttle=set_throttle(currentz,lastz)
            #apply to the vehicle
            vehicle.apply_control(carla.VehicleControl(throttle=input_throttle, steer=input_steer))
            last_steer_esti=steer_esti
            lastz=currentz
        #draw the waypoints in the world
        for w in w1[idx:]:
            w = w[0]
            world.debug.draw_string(w.transform.location, 'o', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=0.2,persistent_lines=True)
        #set delay
        time.sleep(0.02)
    time.sleep(60)

    # for w in w1:
    #     w = w[0]
    #     world.debug.draw_string(w.transform.location, 'o', draw_shadow=False,color=carla.Color(r=255, g=0, b=0), life_time=50.0,persistent_lines=True)
finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')

