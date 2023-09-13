"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qualang_tools.plot import interrupt_on_close
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_vm import *
import numpy as np
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
import cv2
import time


resolution = 1
VP1 = np.arange(0, 115, resolution)
VP2 = np.arange(0, 115, resolution)

main_offset = 0.1

def make_line(start=(), end=()):
    x0 = start[0]; x1 = end[0]
    y0 = start[1]; y1 = end[1]
    slope = (y1-y0) / (x1-x0)
    return start, end, slope

def make_diamond(start=(), diag_step1=0, diag_step2=0, hor_step=0):
    lines = []
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
    return lines

def average_image(current_image, new_image, num_of_frames_till_now):
    if num_of_frames_till_now == 0:
        not_first_image = 0
    else:
        not_first_image = 1
    return ((not_first_image * current_image * num_of_frames_till_now + new_image) / (num_of_frames_till_now + 1))


x0 = 20
y0 = 30

diag1 = 30
diag2 = 15
hor = 5

blur = 1
noise_level = 0.1

step = diag2+2*hor+int(-diag2**2/diag1)
regions = [
    make_diamond((x0, y0), diag1, diag2, hor),
    make_diamond((x0+step, y0+step), diag1, diag2, hor),
    make_diamond((x0+2*step, y0+2*step), diag1, diag2, hor),
    make_diamond((x0+3*step, y0+3*step), diag1, diag2, hor),

    make_diamond((x0+step, y0), diag1, diag2, hor),
    make_diamond((x0+2*step, y0), diag1, diag2, hor),
    make_diamond((x0+3*step, y0), diag1, diag2, hor),
    make_diamond((x0+4*step, y0), diag1, diag2, hor),

    make_diamond((x0+step, y0), diag1, diag2, hor),
    # make_diamond((x0+4*step+hor/2, y0+5*step/2-2*hor), diag1, diag2, hor),
    # make_diamond((x0+3*step+hor/2, y0+3*step/2-2*hor), diag1, diag2, hor),
    # make_diamond((x0+2*step+hor/2, y0+step/2-2*hor), diag1, diag2, hor),
    # make_diamond((x0+1*step+hor/2, y0-step/2-2*hor), diag1, diag2, hor),
]

stability_map = np.zeros((len(VP1), len(VP2)))
for i in range(len(VP1)):
    for j in range(len(VP2)):
        if (5 <= VP1[i] <= 130) and (5 <= VP2[j] <= 130):
            for region in regions:
                for line in region:
                    start, end, slope = line
                    if min(start[0], end[0]) <= VP1[i] <=  max(start[0], end[0]) and min(start[1], end[1]) <= VP2[j] <= max(start[1], end[1]):
                        if np.isclose(VP1[i], start[0] + int(slope * (VP2[j]-start[1]))) or slope == 0:
                            stability_map[i][j] = 1

        else:
            stability_map[i][j] = main_offset

for i in range(len(VP1)):
    for j in range(len(VP2)):
        if stability_map[i][j] == 1 and 0<i<len(VP1) and 0<i<len(VP2):
            for ii in range(1,blur+1):
                for jj in range(1,blur+1):
                    stability_map[i+ii][j+jj] = 1-1/np.sqrt((ii+jj))
                    stability_map[i-ii][j-jj] = 1-1/np.sqrt((ii+jj))
                    stability_map[i+ii][j-jj] = 1-1/np.sqrt((ii+jj))
                    stability_map[i-ii][j+jj] = 1-1/np.sqrt((ii+jj))


for i in range(len(VP1)):
    for j in range(len(VP2)):
        noise = np.random.normal(0, noise_level, 1)
        stability_map[i][j] += noise

plt.figure()
plt.subplot(121)
plt.pcolor(VP2, VP1, stability_map)
plt.axis("equal")
plt.xlabel("VP2")
plt.ylabel("VP1")
plt.title("Python input")


###################
# The QUA program #
###################
# with program() as realtime_prog:
#     vp1 = declare(fixed, value=VP1)
#     vp2 = declare(fixed, value=VP2)
#     start_end0 = declare(int, size=2)
#     start_end1 = declare(int, size=2)
#     path = declare(fixed)
#     start0 = declare(fixed, value=starts0)
#     end0 = declare(fixed, value=ends0)
#     start1 = declare(fixed, value=starts1)
#     end1 = declare(fixed, value=ends0)
#
#
#     I = declare(fixed)
#     a = declare(fixed)
#     i = declare(int)
#     j = declare(int)
#     k = declare(int)
#     I_st = declare_stream()
#     with for_(i, 0, i < len(VP1), i + 1):
#         with for_(j, 0, j < len(VP2), j + 1):
#             assign(start_end0[0], start0)
#             assign(start_end0[1], end0)
#             assign(start_end1[0], start1)
#             assign(start_end1[1], end1)
#             with for_(k, 0, k < len(lines), k + 1):
#
#                 with if_( (Math.min(start_end0) <= vp1[i]) & (Math.max(start_end0) >= vp1[i]) & (Math.min(start_end1) <= vp2[j]) & (Math.max(start_end1) >= vp2[j])):
#                     assign(path,  slope * (vp2[j] - start_end1[0]))
#                     with if_(vp1[i] == path):
#                         assign(a, 1)
#                     with else_():
#                         assign(a, 0)
#             measure("readout"*amp(a), "digitizer", None, integration.full("cos", I, "out1"))
#             save(I, I_st)
#             wait(25)
#     with stream_processing():
#         I_st.buffer(len(VP2)).buffer(len(VP1)).save("map")




#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="172.16.33.100", cluster_name="Cluster_82")
#
# ###########################
# # Run or Simulate Program #
# ###########################

run = True

with program() as hello_qua:
    stab_map = declare(fixed, value=np.concatenate(stability_map))
    r = Random()
    I = declare(fixed)
    a = declare(fixed)
    l = declare(int)
    i = declare(int)
    j = declare(int)
    I_st = declare_stream()
    l_st = declare_stream()
    with for_(l, 0, l < 1000, l + 1):  # repeat for avearging
        save(l, l_st)
        with for_(i, 0, i < len(VP1), i + 1):
            with for_(j, 0, j < len(VP2), j + 1):
                assign(a, stab_map[j + len(VP2) * i])
                measure("readout"*amp(a+r.rand_fixed()*0.5), "digitizer", None, integration.full("cos", I, "out1"))
                save(I, I_st)
                wait(100)
    with stream_processing():
        I_st.buffer(len(VP2)).buffer(len(VP1)).save("map")
        l_st.save("frame_num")

if run:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
    results = fetching_tool(job, data_list=["map", "frame_num"], mode="live")
    # frame_num = []
    # while results.is_processing():
    #     stab_map, frame_num_temp = results.fetch_all()
    #
    # # traditional live plot
    #     frame_num.append(frame_num_temp)
    #     plt.subplot(122)
    #     plt.cla()
    #     plt.pcolor(VP2, VP1, stab_map)
    #     plt.axis("equal")
    #     plt.title(f"OPX acquisition {frame_num[-1]}")
    #     plt.xlabel("VP2")
    #     plt.ylabel("VP1")
    #     plt.tight_layout()
    #     plt.pause(0.1)

# New plotting tool

    enable_avarage = 1
    # Set the desired frame rate (e.g., 10 frames per second)
    desired_frame_rate = 1
    frame_delay = int(1000 / desired_frame_rate)  # Calculate delay in milliseconds

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = [0]  # Use a list to store frame_count as a mutable object
    fps = 0


    # Create a function to generate and return random 100x100 RGB images

    while results.is_processing():
        new_image, frame_num_temp = results.fetch_all()
        if frame_count[0] == 0:
            image = new_image
        if enable_avarage:
            image = average_image(image, new_image, frame_count[0])
        else:
            image = new_image

        # Add FPS text overlay on the image
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(image, fps_text, (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0),
                    1)  # Adjust the position, font, and size

        # Display the image
        cv2.namedWindow("Random Image", cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.imshow("Random Image", image)

        # Calculate FPS
        frame_count[0] += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time != 0:
            fps = frame_count[0] / elapsed_time
        print(elapsed_time)

        # Introduce a delay to achieve the desired frame rate
        if elapsed_time < (frame_count[0] / desired_frame_rate):
            time.sleep((frame_count[0] / desired_frame_rate) - elapsed_time)

        # Press 'q' to quit the stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cv2.destroyAllWindows()
