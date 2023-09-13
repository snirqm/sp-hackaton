from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
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


for diag1 in [20, 40]:
    for diag2 in [10, 15]:
        for hor in [2, 7]:
            for blur in [0, 2]:

                x0 = 20  # position of the first diamond along VP1
                y0 = 50  # position of the first diamond along VP2
                noise_level = 0.25  # Global noise level
                # diag1 = 30  # length of one diamond edge
                # diag2 = 15  # length of the second diamond edge
                # hor = 10  # length of the linking two diamonds
                # blur=1  # width of the transition lines

                # Create the diamonds
                step = diag2 + 2*hor + int(-diag2/diag1*diag2)
                slope = int(-diag2 / diag1 * diag2)
                regions = []
                labels = []
                for j in range(10):
                    for i in range(10):
                        if ((x0 + i*step) < max(VP1)) and ((y0 + i*step) < max(VP2)):
                            x = x0 + i*step + j*(diag2 - slope)
                            y = y0 + i*step + j*(slope - diag2)
                            diamond, label = make_diamond((x, y), diag1, diag2, hor, label=f"({-j+i+1},{i+j+1})")
                            regions.append(diamond)
                            labels.append([x,y,label])

                # Create the stability map
                stability_map = np.zeros((len(VP1), len(VP2)))
                for i in range(len(VP1)):
                    for j in range(len(VP2)):
                        if (5 <= VP1[i] <= 130) and (5 <= VP2[j] <= 130) or True:
                            for region in regions:
                                for line in region:
                                    start, end, slope = line
                                    if min(start[0], end[0]) <= VP1[i] <=  max(start[0], end[0]) and min(start[1], end[1]) <= VP2[j] <= max(start[1], end[1]):
                                        if np.isclose(VP1[i], start[0] + int(slope * (VP2[j]-start[1]))) or slope == 0:
                                            stability_map[i][j] = 1

                        else:
                            stability_map[i][j] = main_offset

                # Broaden the lines and add noise to each mixel
                for i in range(len(VP1)):
                    for j in range(len(VP2)):
                        if stability_map[i][j] == 1 and 0<i<len(VP1)-3 and 0<j<len(VP2)-3:
                            for ii in range(1,blur+1):
                                for jj in range(1,blur+1):
                                    stability_map[i+ii][j+jj] = 1-1/np.sqrt((ii+jj))
                                    stability_map[i-ii][j-jj] = 1-1/np.sqrt((ii+jj))
                                    stability_map[i+ii][j-jj] = 1-1/np.sqrt((ii+jj))
                                    stability_map[i-ii][j+jj] = 1-1/np.sqrt((ii+jj))
                                    noise = np.random.normal(0, noise_level, 1)[0]
                                    stability_map[i][j] += noise
                        else:
                            noise = np.random.normal(0, noise_level, 1)[0]
                            stability_map[i][j] += noise

                plt.figure()
                # plt.subplot(121)
                plt.pcolor(VP2, VP1, stability_map)
                plt.axis("equal")
                plt.xlabel("VP2")
                plt.ylabel("VP1")
                plt.title("Python input")
                for ll in labels:
                    if ll[1] > VP2[0] and ll[0]<VP1[-1]:
                        plt.text(ll[1]+2, ll[0]+2, ll[2], bbox=dict(facecolor='red', alpha=0.5))


###################
# The QUA program #
###################
run = False

if run:
    qmm = QuantumMachinesManager(host="192.168.0.129", cluster_name="Cluster_1")
    with program() as hello_qua:
        stab_map = declare(fixed, value=np.concatenate(stability_map))
        I = declare(fixed)
        a = declare(fixed)
        i = declare(int)
        j = declare(int)
        I_st = declare_stream()
        with for_(i, 0, i < len(VP1), i + 1):
            with for_(j, 0, j < len(VP2), j + 1):
                assign(a, stab_map[j + len(VP2) * i])
                measure("readout" * amp(a), "digitizer", None, integration.full("cos", I, "out1"))
                save(I, I_st)
                wait(25)
        with stream_processing():
            I_st.buffer(len(VP2)).buffer(len(VP1)).save("map")
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
    job.result_handles.wait_for_all_values()
    stab_map = job.result_handles.get("map").fetch_all()
    plt.subplot(122)
    plt.pcolor(VP2, VP1, stab_map)
    plt.axis("equal")
    plt.title("OPX acquisition")
    plt.xlabel("VP2")
    plt.ylabel("VP1")

    plt.tight_layout()

