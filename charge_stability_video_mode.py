"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qualang_tools.plot import interrupt_on_close
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import numpy as np
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
import cv2
import time
import warnings

warnings.filterwarnings("ignore")


def average_image(current_image, new_image, num_of_frames_till_now):
    if num_of_frames_till_now == 0:
        not_first_image = 0
    else:
        not_first_image = 1
    return ((not_first_image * current_image * num_of_frames_till_now + new_image) / (num_of_frames_till_now + 1))


resolution = 1
VP1 = np.arange(0, 115, resolution)
VP2 = np.arange(0, 115, resolution)

main_offset = 0.1

def make_line(start=(), end=()):
    x0 = start[0]; x1 = end[0]
    y0 = start[1]; y1 = end[1]
    slope = (y1-y0) / (x1-x0)
    return start, end, slope

def make_diamond(start=(), diag_step1=0, diag_step2=0, hor_step=0, label="(0,0)"):
    x0 = start[0]; y0 = start[1]
    slope = -int(diag_step2 / diag_step1 * diag_step2)
    lines = [make_line((x0, y0), (x0 - diag_step1, y0 + diag_step2)),
             make_line((x0, y0), (x0 + diag_step2, y0 - diag_step1)),
             make_line((x0 + slope, y0 + diag_step2), (x0 + slope +hor_step, y0 + diag_step2+hor_step)),
             make_line((x0 + diag_step2, y0 + slope), (x0 + diag_step2 + hor_step, y0 + slope + hor_step)),
             make_line((x0 + slope + hor_step, y0 + diag_step2 + hor_step), (x0 + slope + hor_step + diag_step2, y0 + diag_step2 + hor_step - diag_step1)),
             make_line((x0 + diag_step2 + hor_step, y0 + slope+hor_step), (x0 + diag_step2+hor_step - diag_step1, y0 + slope+hor_step + diag_step2)),
             make_line((x0 + diag_step2 + hor_step + slope, y0 + slope+hor_step + diag_step2), (x0 + diag_step2 + 2*hor_step + slope, y0 + slope+hor_step + diag_step2+hor_step)),
             ]
    return lines, label



for device in range(1):
    diag1 = int(np.random.uniform(40, 80))
    diag2 = int(np.random.uniform(15, diag1/2))
    hor = int(np.random.uniform(1, diag2/2))
    blur = int(np.random.uniform(1, 4))
    for realization in range(1):
        x0 = int(np.random.uniform(10, 50))
        y0 = int(np.random.uniform(15, 65))
        noise_level = np.random.uniform(0.0, 0.5)
        parameters = f"diag1_{diag1}_diag2_{diag2}_hor_{hor}_width_{blur}_noise_{int(noise_level*1000)}_x0_{x0}_y0_{y0}"
        # x0 = 20  # position of the first diamond along VP1
        # y0 = 50  # position of the first diamond along VP2
        # noise_level = 0.25  # Global noise level
        # diag1 = 30  # length of one diamond edge
        # diag2 = 15  # length of the second diamond edge
        # hor = 10  # length of the linking two diamonds
        # blur=1  # width of the transition lines
        start_time = time.time()
        # Create the diamonds
        slope = int(-diag2 / diag1 * diag2)
        step = diag2 + 2*hor + slope
        regions = []
        labels = []
        for j in range(-1,10):
            for i in range(10):
                if ((x0 + i*step) < max(VP1)) and ((y0 + i*step) < max(VP2)):
                    x = x0 + i*step + j*(diag2 - slope)
                    y = y0 + i*step + j*(slope - diag2)
                    if (j+i) >= 0 and (i-j+1) >= 0:
                        diamond, label = make_diamond((x, y), diag1, diag2, hor, label=f"({-j+i+1},{i+j+1})")
                        labels.append([x + (step - hor) / 2, y + (step - hor) / 2, f"({j + i},{i - j + 1})"])
                        regions.append(diamond)
                    if (j+i) >= 0 and (i-j+2) >= 0:
                        diamond2, label2 = make_diamond((x + step/2 + (slope-diag2)/2, y + step/2 - (slope-diag2)/2), diag1, diag2, hor, label=f"({-j+i+2},{i+j+1})")
                        labels.append([x + (step - hor) / 2 + step / 2 + (slope - diag2) / 2,
                                       y + (step - hor) / 2 + step / 2 - (slope - diag2) / 2,
                                       f"({j + i},{i - j + 2})"])
                        regions.append(diamond2)


        # Create the stability map
        stability_map = np.zeros((len(VP1), len(VP2)))
        bit_map = np.zeros((len(VP1), len(VP2)))

        diamond_counter = 0
        for region in regions:
            bit_map1 = np.zeros((len(VP1), len(VP2)))
            line_counter = 0
            label = labels[diamond_counter]
            diamond_counter += 1
            for line in region:
                line_counter += 1
                start, end, slope = line
                for i in range(len(VP1)):
                    for j in range(len(VP2)):
                        if min(start[0], end[0]) <= VP1[i] <= max(start[0], end[0]) and min(start[1], end[1]) <= VP2[
                            j] <= max(start[1], end[1]):
                            if np.isclose(VP1[i], start[0] + int(slope * (VP2[j] - start[1]))) or slope == 0:
                                stability_map[i][j] = 1
                                bit_map[i][j] = 1
                                if line_counter < len(region):
                                    bit_map1[i][j] = 1

        # Broaden the lines and add noise to each mixel
        for i in range(len(VP1)):
            for j in range(len(VP2)):
                if stability_map[i][j] == 1 and 0<i<len(VP1)-3 and 0<j<len(VP2)-3:
                    for ii in range(0,blur):
                        for jj in range(0,blur):
                            if ii+jj == 0:
                                stability_map[i+ii][j+jj] = 1
                            else:
                                stability_map[i+ii][j+jj] = 1-(ii+jj)/4
                                stability_map[i-ii][j-jj] = 1-(ii+jj)/4
                                stability_map[i+ii][j-jj] = 1-(ii+jj)/4
                                stability_map[i-ii][j+jj] = 1-(ii+jj)/4
                            noise = np.random.normal(0, noise_level, 1)[0]
                            stability_map[i][j] += noise
                else:
                    noise = np.random.normal(0, noise_level, 1)[0]
                    stability_map[i][j] += noise


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="192.168.0.129", cluster_name="Cluster_1")
#
# ###########################
# # Run or Simulate Program #
# ###########################

run = True

with program() as hello_qua:
    stab_map = declare(fixed, value=np.concatenate(stability_map))
    r = Random()
    dx = declare(int)
    dy = declare(int)
    I = declare(fixed)
    a = declare(fixed)
    l = declare(int)
    i = declare(int)
    j = declare(int)
    I_st = declare_stream()
    l_st = declare_stream()
    with for_(l, 0, l < 100000, l + 1):  # repeat for averaging
        save(l, l_st)
        with for_(i, 0, i < len(VP1), i + 1):
            with for_(j, 0, j < len(VP2), j + 1):
                assign(a, stab_map[j + len(VP2) * i])
                measure("readout"*amp(a+r.rand_fixed()*0.2), "digitizer", None, integration.full("cos", I, "out1"))
                save(I, I_st)
                wait(400//4)
    with stream_processing():
        I_st.buffer(len(VP2)).buffer(len(VP1)).save("map")
        l_st.save("frame_num")

if run:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
    results = fetching_tool(job, data_list=["map", "frame_num"], mode="live")
    frame_num = []
    start = time.time()
    count = 0
    frame_num_prev = 0
    while results.is_processing():
        count += 1
        stab_map, frame_num_temp = results.fetch_all()
        if frame_num_prev < frame_num_temp:
            plt.cla()
            plt.pcolor(VP2, VP1, stab_map)
            plt.axis("equal")
            plt.title(f"plotting: {count/(time.time()-start):.1f} fps, streaming: {frame_num_temp/(time.time()-start):.1f} fps")
            plt.xlabel("VP2")
            plt.ylabel("VP1")
            plt.tight_layout()
            plt.pause(0.001)
        frame_num_prev = frame_num_temp

# New plotting tool
#
#     enable_avarage = 1
#     # Set the desired frame rate (e.g., 10 frames per second)
#     desired_frame_rate = 1
#     frame_delay = int(1000 / desired_frame_rate)  # Calculate delay in milliseconds
#
#     # Initialize variables for FPS calculation
#     start_time = time.time()
#     frame_count = [0]  # Use a list to store frame_count as a mutable object
#     fps = 0


    # Create a function to generate and return random 100x100 RGB images
    #
    # while results.is_processing():
    #     new_image, frame_num_temp = results.fetch_all()
    #     if frame_count[0] == 0:
    #         image = new_image
    #     if enable_avarage:
    #         image = average_image(image, new_image, frame_count[0])
    #     else:
    #         image = new_image
    #
    #     # Add FPS text overlay on the image
    #     fps_text = f"FPS: {fps:.2f}"
    #     cv2.putText(image, fps_text, (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0),
    #                 1)  # Adjust the position, font, and size
    #
    #     # Display the image
    #     cv2.namedWindow("Random Image", cv2.WINDOW_NORMAL)  # Create a resizable window
    #     cv2.imshow("Random Image", image)
    #
    #     # Calculate FPS
    #     frame_count[0] += 1
    #     current_time = time.time()
    #     elapsed_time = current_time - start_time
    #     if elapsed_time != 0:
    #         fps = frame_count[0] / elapsed_time
    #     print(elapsed_time)
    #
    #     # Introduce a delay to achieve the desired frame rate
    #     if elapsed_time < (frame_count[0] / desired_frame_rate):
    #         time.sleep((frame_count[0] / desired_frame_rate) - elapsed_time)
    #
    #     # Press 'q' to quit the stream
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # Release the VideoCapture and close all OpenCV windows
    # cv2.destroyAllWindows()
