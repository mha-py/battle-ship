import cv2
import numpy as np
import matplotlib.pyplot as plt

class vidcapture_cv2:
    def __init__(self, filename, fps, resolution_x, resolution_y, DPI_CONST=100):
        self.DPI_CONST = DPI_CONST  # dient als Umrechenfaktor zwischen Plotgröße und Bildgröße,
                                    # nur bei Übergabe des Frames durch 'plt' nötig.
        self.figsize = (resolution_x/self.DPI_CONST, resolution_y/self.DPI_CONST)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (resolution_x, resolution_y), isColor=True)
        
    def figure(self):
        '''Erstellt eine plt-Figure. Wrappt den Befehl 'plt.figure()' und übergibt die richtige Bildgröße.'''
        plt.figure(figsize=self.figsize)
        
    def capture(self, frame=None):
        '''Zwei Varianten sind möglich: Ohne Angabe von 'frame' muss der Frame geplottet werden mit der figsize, 
        die durch capture_begin ausgegeben wurde. Sonst kann der Frame als np.array als float oder int im RGB-Format
        übergeben. Beachte, dass in einem Array zuerst der Zeilenindex, dann der Spaltenindex kommt, was vertauscht ist
        gegenüber der x- und y-Koordinate.'''
        if frame is None:
            global plt
            plt.savefig('temp.png', dpi=self.DPI_CONST)
            frame = cv2.imread('temp.png', 1)
        else:
            frame = frame[:,:,::-1] # RGB to BGR
            if frame.dtype == 'float':
                frame = (255*frame).astype('uint8') # float to int
        self.writer.write(frame)
        
    def close(self):
        self.writer.release()
        
        
        
import os

class vidcapture:
    '''Die .bat Datei zum Kompilieren lautet, wenn filename='frame%%05d.png':
    ffmpeg -r 10 -f image2 -i frame%%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p video.mp4
    @ping -n 10000 localhost> nul'''
    def __init__(self, filename, resolution_x=200, resolution_y=200, DPI_CONST=100):
        self.DPI_CONST = DPI_CONST  # dient als Umrechenfaktor zwischen Plotgröße und Bildgröße,
                                    # nur bei Übergabe des Frames durch 'plt' nötig.
        self.resx = resolution_x
        self.resy = resolution_y
        self.filename = filename
        self.i = 0
        
        # Ordner erstellen für die Bilder
        try:
            dirname = os.path.dirname(filename)
            os.mkdir(dirname)
        except:
            pass
        
    def figure(self, image_like=True):
        '''Erstellt eine plt-Figure. Wrappt den Befehl 'plt.figure()' und übergibt die richtige Bildgröße.'''
        fig = plt.figure(figsize=(self.resx/self.DPI_CONST, self.resy/self.DPI_CONST))
        if image_like:
            fig.subplots_adjust(bottom = 0)
            fig.subplots_adjust(top = 1)
            fig.subplots_adjust(left = 0)
            fig.subplots_adjust(right = 1)
            plt.axis('off')
        return fig
        
    def capture(self, frame=None):
        '''Zwei Varianten sind möglich: Ohne Angabe von 'frame' wird der plot als Frame gespeichert mit der figsize.
        Sonst kann der Frame als np.array mit dtype float oder int im RGB-Format übergeben. Beachte, dass in einem
        Array zuerst der Zeilenindex, dann der Spaltenindex kommt, was vertauscht ist gegenüber der x- und y-Koordinate.'''
        if frame is None:
            plt.savefig(self.filename % self.i, dpi=self.DPI_CONST)
            self.i += 1
        else:
            plt.imsave(self.filename % self.i, frame)
            self.i += 1
            