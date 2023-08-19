import numpy as np

class ra_encoder():
    def __init__(self, geometry, statistics,subpixel,regression_layer = 2):
        
        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer
        self.calib = np.load('./camera_calib.npy',allow_pickle=True).item()

        self.INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        self.subpixel = subpixel
        self.OUTPUT_DIM = (regression_layer + 1,self.INPUT_DIM[0] // self.subpixel , self.INPUT_DIM[1] // self.subpixel )

    def decode_points(self, pcl_points, pred_obj):
        Coord = []
        Coord_RA = []
        for i in range(len(pred_obj)):
            coordinates = []
            coordinates_RA = []

            for bin in range(len(pred_obj[i])):
                pred_rcls_bins = pred_obj[i][bin,:11]
                pred_rcls_idx = int(pred_rcls_bins.argsort()[-1])

                rc = (pcl_points[i][bin,0]/0.201171875/2).astype(int)
                R_center = int(rc-(pred_rcls_idx-5))
                R = float(R_center*self.subpixel*self.geometry['resolution'][0]+pred_obj[i][bin,16]* self.statistics['reg_std'][0] + self.statistics['reg_mean'][0])


                pred_acls_bins = pred_obj[i][bin,11:16]
                pred_acls_idx = int(pred_acls_bins.argsort()[-1])

                ac = np.clip(np.floor(pcl_points[i][bin,1]/0.2/2 + 224),0,448).astype(int)
                A_center = int(ac-(pred_acls_idx-2))
                A = float((A_center-self.OUTPUT_DIM[2]/2)*self.subpixel*self.geometry['resolution'][1] + pred_obj[i][bin,17] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1])

                if [R,A] not in coordinates:
                    coordinates.append([R,A])
                    coordinates_RA.append([R_center,A_center,float(pred_obj[i][bin,16]),float(pred_obj[i][bin,17])])
            Coord.append(np.asarray(coordinates))
            Coord_RA.append(np.asarray(coordinates_RA))
        return Coord,Coord_RA


    def filter(self,pcls, box_labels,detector):
        PCL = []
        L = 4

        for label,pcl in zip(box_labels,pcls):
            x_mean = int((label[3]+label[5])/2)
            y_mean = int((label[4]+label[6])/2)

            x_radius = int((label[5]-label[3])/2) 
            y_radius = int((label[6]-label[4])/2)

            if detector == 'gt':
                r = np.min([x_radius,y_radius])/2
                x_mean = x_mean/2
                y_mean = y_mean/2
            else:
                r = np.min([x_radius,y_radius])

            id_instance = np.where( (pcl[:,3]>=(x_mean-r)) & (pcl[:,3]<=(x_mean+r)) & (pcl[:,4]>=(y_mean-r)) & (pcl[:,4]<=(y_mean+r)) )[0]
            if len(id_instance) == 0:
                id_instance = np.arange(len(pcl))
            
            idx = np.argmax(pcl[id_instance,4])
            instance = id_instance[idx]

            r = pcl[instance,0]
            # phi = pcl[instance,1]

            id = np.where((pcl[:,0]<(r+L)))[0]

            pcl = pcl[id]
            PCL.append(pcl)
        return PCL