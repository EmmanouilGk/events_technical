import sys
from matplotlib import pyplot as plt
import networkx as nx
import logging
logging.basicConfig(filename="/home/iccs/Desktop/isense/events/intention_prediction/utils/logs.log")
logging.getLogger("draw_module")

def draw_g(G_1:nx.Graph ,img ):
    #ex1
    G = nx.petersen_graph()
    fig1 = plt.figure(0)
    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    fig1.savefig("/home/iccs/Desktop/isense/events/intention_prediction/utils/tmp/petersong.png")

    fig1.savefig("/home/iccs/Desktop/isense/events/intention_prediction/utils/tmp/petersong2.png")

    #our
    fig2 = plt.figure(2)
    nx.draw_networkx(G_1 , with_labels = True , font_weight = "bold" , arrows=True)
    fig2.savefig("/home/iccs/Desktop/isense/events/intention_prediction/utils/tmp/prevention_G.png")

    #overlaid
    fig3 = plt.figure(4,figsize=(18,10))

    plt.imshow(img)

    #with pos
    logging.info("Graph G of highway traffic has {} nodes".format(G_1.nodes.data()))
    print("Graph G has the att/te degree statistics {} and connected components \n and degrees {}".format(nx.clustering(G_1) , list(nx.connected_components(G_1)) , sorted(d for n, d in G.degree())))
    # with open('/home/iccs/Desktop/isense/events/intention_prediction/utils/logs.log', 'w') as sys.stdout:
    #        print("Graph G has the att/te degree statistics {} and connected components \n and degrees {}".format(nx.clustering(G_1) , list(nx.connected_components(G_1)) , sorted(d for n, d in G.degree())))

    nx.draw_networkx(G_1 , pos={k:v["loc"] for k,v in G_1.nodes.data()} ,  with_labels = True ,arrows=True  ,arrowsize = 15)
    fig3.savefig("/home/iccs/Desktop/isense/events/intention_prediction/utils/tmp/highway_fram_with_graph.png")



    
