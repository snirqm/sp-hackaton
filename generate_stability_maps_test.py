from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
# from configuration import *
import time
import numpy as np
import matplotlib.pyplot as plt


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


for diag1 in [40, 45]:
    for diag2 in [20, 25]:
        for hor in [3,7]:
            for blur in [1,3]:
                for noise_level in [0.25, 0.5]:
                    for x0 in [20, 40]:
                        for y0 in [50, 60]:
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

                            # plt.figure()
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
                                if len(np.where(np.concatenate(bit_map1)>0)[0]) > 0:
                                    plt.figure()
                                    plt.pcolor(bit_map1)
                                    plt.axis("equal")
                                    plt.axis('off')
                                    plt.savefig(f"./training_dataset/{parameters}_bitmap_{label[2][1]}_{label[2][3]}")
                                    plt.close()

                            # Broaden the lines and add noise to each mixel
                            for i in range(len(VP1)):
                                for j in range(len(VP2)):
                                    if stability_map[i][j] == 1 and 0<i<len(VP1)-3 and 0<j<len(VP2)-3:
                                        for ii in range(0,blur):
                                            for jj in range(0,blur):
                                                if ii+jj == 0:
                                                    stability_map[i+ii][j+jj] = 1/np.sqrt(blur+1)
                                                else:
                                                    stability_map[i+ii][j+jj] = 1-1/np.sqrt((0.1+ii+jj))
                                                    stability_map[i-ii][j-jj] = 1-1/np.sqrt((0.1+ii+jj))
                                                    stability_map[i+ii][j-jj] = 1-1/np.sqrt((0.1+ii+jj))
                                                    stability_map[i-ii][j+jj] = 1-1/np.sqrt((0.1+ii+jj))
                                                noise = np.random.normal(0, noise_level, 1)[0]
                                                stability_map[i][j] += noise
                                    else:
                                        noise = np.random.normal(0, noise_level, 1)[0]
                                        stability_map[i][j] += noise
                            plt.figure()
                            plt.pcolor(stability_map)
                            plt.axis("equal")
                            plt.axis('off')
                            plt.savefig(f"./training_dataset/{parameters}_image")
                            plt.close()


                            plt.figure()
                            plt.subplot(121)
                            plt.pcolor(VP2, VP1, stability_map)
                            plt.axis("equal")
                            plt.title("Image")
                            plt.axis('off')
                            plt.subplot(122)
                            plt.pcolor(VP2, VP1, bit_map)
                            plt.axis("equal")
                            plt.title("Bitmap")
                            plt.axis('off')
                            for ll in labels:
                                if VP2[-1] > ll[1] > VP2[0] and VP1[0] < ll[0] < VP1[-1]:
                                    plt.text(ll[1], ll[0], ll[2], bbox=dict(facecolor='red', alpha=0.5))
                                    plt.plot(ll[1], ll[0], "r*")
                            plt.savefig(f"./training_dataset/{parameters}_full_image.jpeg")
                            plt.close()
                            print(f"elapsed time: {time.time()-start_time:.0f}s")


###################
# The QUA program #
###################
run = False

# if run:
#     qmm = QuantumMachinesManager(host="192.168.0.129", cluster_name="Cluster_1")
#     with program() as hello_qua:
#         stab_map = declare(fixed, value=np.concatenate(stability_map))
#         I = declare(fixed)
#         a = declare(fixed)
#         i = declare(int)
#         j = declare(int)
#         I_st = declare_stream()
#         with for_(i, 0, i < len(VP1), i + 1):
#             with for_(j, 0, j < len(VP2), j + 1):
#                 assign(a, stab_map[j + len(VP2) * i])
#                 measure("readout" * amp(a), "digitizer", None, integration.full("cos", I, "out1"))
#                 save(I, I_st)
#                 wait(25)
#         with stream_processing():
#             I_st.buffer(len(VP2)).buffer(len(VP1)).save("map")
#     # Open a quantum machine to execute the QUA program
#     qm = qmm.open_qm(config)
#     # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
#     job = qm.execute(hello_qua)
#     job.result_handles.wait_for_all_values()
#     stab_map = job.result_handles.get("map").fetch_all()
#     plt.subplot(122)
#     plt.pcolor(VP2, VP1, stab_map)
#     plt.axis("equal")
#     plt.title("OPX acquisition")
#     plt.xlabel("VP2")
#     plt.ylabel("VP1")
#
#     plt.tight_layout()

