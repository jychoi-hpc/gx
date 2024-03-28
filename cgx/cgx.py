# -*- coding: utf-8 -*-
"""
Originally created by Jake Bringewatt on Thu Jul 28 09:34:53 2016
GUI to Create Various Plots from NetCDF Output Files

Required python libraries:
-PyQt4
-netCDF 
-matplotlib
-numpy

@author: Jake Bringewatt

Extended and maintained by Bill Dorland

"""

from PyQt4.uic import loadUiType
from netCDF4 import Dataset
from netCDF4 import MFDataset
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.colors import Colormap
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

qtCreatorFile = "w_gui.ui" # Enter file here.

Ui_MainWindow, QMainWindow = loadUiType(qtCreatorFile)

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, files):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}
        self.slope_dict={}
        self.dict_id=0
        
        #Define actions for button clicks
        self.chooseFileButton.clicked.connect(self.FileDialog)
        self.selectAllButton.clicked.connect(self.selectAllPlots)
        self.updatePlotsButton.clicked.connect(self.updatePlots)
        self.createPlotsButton.clicked.connect(self.createPlots)
        self.figList.itemClicked.connect(self.changefig)
        self.combineDataCheckBox.clicked.connect(self.combineDataOption)
        self.selectedDataCheckBox.clicked.connect(self.selectedDataOption)
        
        #define file list        
        self.files=files

        #if files were input in command line add those files to the file list text box
        if(self.files):            
            self.fileListDisplay.insertItems(-1, self.files) 
            self.fileListDisplay.setCurrentRow(0)

        #Add empty placeholder figure
        fig = Figure()
        self.addmpl(fig)

        #define average and data combine type variables (used in naming figures)
        self.averageType=''
        self.dataCombineType=''
        
    def selectAllPlots(self,):
         self.plotKSpecFixedMCheckBox.setChecked(True)
         self.plotMSpecFixedKCheckBox.setChecked(True)
         self.plot2DG2CheckBox.setChecked(False)
         #self.plotKSpecVsTimeCheckBox.setChecked(False)
         #self.plotMSpecVsTimeCheckBox.setChecked(False)
         self.plotKSpecCheckBox.setChecked(True)
         self.plotMSpecCheckBox.setChecked(True)
         self.Phi2VsTimeCheckBox.setChecked(True)
         self.E2VsTimeCheckBox.setChecked(True)
        
    def combineDataOption(self,):
        self.selectedDataCheckBox.setChecked(False)
    
    def selectedDataOption(self,):
        self.combineDataCheckBox.setChecked(False)        
        
    def updatePlots(self,):
        for key, value in self.fig_dict.iteritems():
            currFig=value
            for obj in currFig.findobj():
                if obj.get_label()=='plabel':
                    if(self.MSlopeCheckBox.isChecked()):                          
                        obj.set_visible(True)                            
                    elif(not self.MSlopeCheckBox.isChecked()):                      
                        obj.set_visible(False)     
            axes=currFig.get_axes()
            for axis in axes:
                if(self.gridLinesMajorCheckBox.isChecked()):                
                    axis.grid(b=True, which='major', linestyle='-')
                elif(not self.gridLinesMajorCheckBox.isChecked()):
                    axis.grid(b=False, which='major')
                if self.gridLinesMinorCheckBox.isChecked():                    
                    axis.grid(b=True, which='minor', linestyle='-', color='grey')
                elif (not self.gridLinesMinorCheckBox.isChecked()):
                    axis.grid(b=False, which='minor')
            
               #remove legend
                legend=axis.get_legend()
                if(not legend==None):
                    legend.remove()
                    
                #reset legend handles
                legendHandles=[]
                
                #Set data visibility appropriately
                for obj in axis.findobj():                   
                    if obj.get_label()=='Kinetic Energy':
                        if(self.G2CheckBox.isChecked()):                          
                            obj.set_visible(True)
                            legendHandles.append(obj)
                        elif(not self.G2CheckBox.isChecked()):                      
                            obj.set_visible(False)
                    if obj.get_label()=='Fit':
                        if(self.MSlopeCheckBox.isChecked()):                          
                            obj.set_visible(True)
                            legendHandles.append(obj)
                        elif(not self.MSlopeCheckBox.isChecked()):                      
                            obj.set_visible(False)
                            
                #Re-add legend if needed                
                if(self.legendCheckBox.isChecked() and len(legendHandles)>0):                                        
                    axis.legend(handles=legendHandles, loc=1)                 
                        
        #redraw current plot
        currItem=self.figList.currentItem()
        if not currItem==None:
            self.rmmpl()
            self.addmpl(self.fig_dict[currItem.text()])


    def FileDialog(self,):
        newFiles= QtGui.QFileDialog.getOpenFileNames(self, 'OpenFile', os.getcwd(), 'NetCDF Files (*.nc)')
        self.files+=newFiles        
        self.fileListDisplay.insertItems(-1, newFiles)
        self.fileListDisplay.setCurrentRow(0)
               
    def createPlots(self,):
        #get data
        #get data and combine along time (unlimited) dimension
        if(self.combineDataCheckBox.isChecked()):           
            fh=MFDataset(self.files)
            self.dataCombineType='combined_'
            
        #Get data only from selected data file
        elif(self.selectedDataCheckBox.isChecked()): 
            myfile=self.fileListDisplay.currentItem().text()
            fh=Dataset(myfile, mode='r')
            self.dataCombineType='file_'+str(self.fileListDisplay.currentRow())+'_'
            
        #read data
        t=fh.variables['t'][:]
        kpar=fh.variables['ky'][:]        
        mm=fh.variables['mm'][:]
        Phi2_k=fh.variables['Phi2_by_k'][:]
        G2_mk=fh.variables['G2k'][:]
        Gth_mk=fh.variables['Gthk'][:]
        Gth_rotated_mk=fh.variables['Rotated_Gthk'][:]
        #C_diss_mk=fh.variables['C_diss'][:]
        #HP_diss=fh.variables['HP_diss'][:]
        fh.close()
        
        #Get time average or standard average (or time integral?) BD
        if(len(t)==1):
            myweights=np.ones(1)
        else:               
            myweights=np.ones(len(t)-1) 
        self.averageType='stdAvg_'
        
        #If time weighted change averaging weights based on time differences
        if self.averageTypeComboBox.currentIndex()==0:
            
            #get time weights         
            self.averageType='tAvg_'
            
            #Check for single time slice case
            if(len(t)==1):
                myweights[0]=1
                startIdx=0
            else:
                for i in range(1,len(t)):                  
                    val= t[i]-t[i-1]
                    myweights[i-1]=val 
                startIdx=1
                
        G2DataFinal=np.zeros((len(mm),len(kpar)))
        P2DataFinal=np.zeros(len(kpar))
        
        for mIdx in range(len(mm)):
            for kparIdx in range(len(kpar)):
                G2=np.average(G2_mk[startIdx:, mIdx, kparIdx], weights=myweights)                
                G2DataFinal[mIdx, kparIdx]=G2

        for kparIdx in range(len(kpar)):
            P2=np.average(Phi2_k[startIdx:, kparIdx], weights=myweights)
            P2DataFinal[kparIdx]=P2
            
        #Create desired plots
        if self.plotKSpecFixedMCheckBox.isChecked():
            self.createKSpecFixedM(mm, kpar, G2DataFinal)

        if self.plotMSpecFixedKCheckBox.isChecked():
            self.createMSpecFixedK(mm, kpar, G2DataFinal)

        if self.plotMSpecCheckBox.isChecked():
            self.createMSpec(mm, kpar, G2DataFinal)
            
        if self.plotKSpecCheckBox.isChecked():
            self.createKSpec(mm, kpar, G2DataFinal)
            
        if self.plotMSpecVsTimeCheckBox.isChecked():
            self.createMSpecVsTime(t, mm, kpar, G2_mk)
            
        if self.plotKSpecVsTimeCheckBox.isChecked():
            self.createKSpecVsTime(t, mm, kpar, G2_mk)
            
        if self.plot2DG2CheckBox.isChecked():
            self.create2DSpectra(mm, kpar, G2DataFinal, 0) 
            
        if self.Phi2VsTimeCheckBox.isChecked():
            self.createPhi2VsTime(t, kpar, Phi2_k)
            
        if self.E2VsTimeCheckBox.isChecked():
            self.createE2VsTime(t, kpar, Phi2_k)
            
        #Add appropriate plot settings to new plots
        self.updatePlots()        
     
    def createPhi2VsTime(self, t, kpar, P2data):     
        #integrate over kpar and kperp to get energy at each timestep
        P2Final=np.zeros(len(t)) 
        for i in range(len(t)):
            valP2=0
            for j in range(len(kpar)):
                valP2+=P2data[i, j]
            P2Final[i]=valP2

        Phi2VsTimeFig=Figure()
        Phi2VsTimeAxis=Phi2VsTimeFig.add_subplot(111)        
        P2Data =Phi2VsTimeAxis.plot(t, P2Final, color='blue',  label='$\Phi^2$', linewidth=2)  
        Phi2VsTimeAxis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        Phi2VsTimeAxis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())        
        Phi2VsTimeAxis.set_title('$1/L \, \int \Phi^2$ vs Time')
        Phi2VsTimeAxis.set_xlabel('Time')
        Phi2VsTimeAxis.set_ylabel('$1/L \, \int \Phi^2$')
        
        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        Phi2VsTimeFig.text(0.05,.95,label)
        
        self.addfig('Phi^2 vs Time_'+self.averageType+self.dataCombineType+str(self.dict_id), Phi2VsTimeFig)
          
    def createE2VsTime(self, t, kpar, P2data):     
        E2Final=np.zeros(len(t)) 
        for i in range(len(t)):
            valE2=0
            for j in range(len(kpar)):
                valE2+=P2data[i, j]*kpar[j]**2
            E2Final[i]=valE2

        E2VsTimeFig=Figure()
        E2VsTimeAxis=E2VsTimeFig.add_subplot(111)        
        E2Data =E2VsTimeAxis.plot(t, E2Final, color='blue',  label='$E^2$', linewidth=2)  
        E2VsTimeAxis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        E2VsTimeAxis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())        
        E2VsTimeAxis.set_title('$1/L \, \int E^2$ vs Time')
        E2VsTimeAxis.set_xlabel('Time')
        E2VsTimeAxis.set_ylabel('$1/L \, \int E^2$')
        
        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        E2VsTimeFig.text(0.05,.95,label)
        
        self.addfig('E^2 vs Time_'+self.averageType+self.dataCombineType+str(self.dict_id), E2VsTimeFig)
          
    def createKSpec(self, mm, kpar, G2DataFinal):
        #Create a kpar spectrum integrated over m
        G2DataFinalInt=np.zeros(len(kpar))
        for i in range(len(kpar)):
            G2DataFinalInt[i]=np.sum(G2DataFinal[:, i])
            
        #Name the figure  
        figName='k spectrum _'+self.averageType+self.dataCombineType+str(self.dict_id)
        
        #Create figure
        KSpecFig = Figure()            
        KSpecAxis =  KSpecFig.add_subplot(111)
        KSpecAxis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        KSpecAxis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        KSpecAxis.set_yscale('log')
        KSpecAxis.set_xscale('log')
        
        #Mask values where energy is small
        kpar = np.ma.masked_where(G2DataFinalInt<=1.e-16, kpar)        
        G2DataFinalInt = np.ma.masked_where(G2DataFinalInt<=1.e-16, G2DataFinalInt)

        KSpecAxis.scatter(kpar[:],G2DataFinalInt[:], color='blue',s=30,edgecolor='none', label='$G^2$')  
        KSpecAxis.set_title('$k_\parallel$ Spectrum summed over $m$')
        KSpecAxis.set_xlabel('$k_\parallel$')
        KSpecAxis.set_ylabel('$G^2$') 
                    
        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        KSpecFig.text(0.05,.95,label)
                                
        self.addfig(figName, KSpecFig)

    def createMSpec(self, mm, kpar, G2DataFinal):
        #Create an m spectrum integrated over kpar
        G2DataFinalInt=np.zeros(len(mm))
        mp=np.zeros(len(mm))
        for i in range(len(mm)):
            G2DataFinalInt[i]=np.sum(G2DataFinal[i, :])
            mp[i]=mm[i]+1
            
        #Name the figure
        figName='m spectrum _'+self.averageType+self.dataCombineType+str(self.dict_id)
        
        #Create figure
        MSpecFig = Figure()            
        MSpecAxis = MSpecFig.add_subplot(111)
        MSpecAxis.set_yscale('log')
        MSpecAxis.set_xscale('log')

        MSpecAxis.scatter(mp[:],G2DataFinalInt[:], color='blue',s=30,edgecolor='none', label='$G^2$')  
        
        #get the slope of the total spectrum
        slope, c=self.getSlope(np.log(mp), np.log(G2DataFinalInt))
        
        #add the slope to the slope dictionary
        self.slope_dict[figName]=slope 
        
        #Create k^(slope) line        
        x = np.arange(mp[1], mp[-1], 1)
        y = (2.718**c)*x**(slope)
        MSpecAxis.plot(x,y,label='Fit')
        
        #Create slope label
        MSpecFig.text(0.80, 0.95, "p="+'%.5f' % (-1*slope), label="plabel")
        
        MSpecAxis.set_title('$m$ spectrum')
        MSpecAxis.set_xlabel('$m+1$')
        MSpecAxis.set_ylabel('$G^2$')      

        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        MSpecFig.text(0.05,0.95,label)
                     
        self.addfig(figName, MSpecFig)  
        
    def createKSpecVsTime(self, t, mm, kpar, dataFinal):
        dataFinalInt=np.zeros((len(t),len(kpar)))
        for i in range(len(kpar)):
            for j in range(len(t)):
                dataFinalInt[j, i]=np.sum(dataFinal[j, :, i])

        tgrid, kgrid = np.meshgrid(t, kpar)
        KSpecVsTimeFig = Figure()
        KSpecVsTimeAxis = KSpecVsTimeFig.add_subplot(111)

        dataWithoutZeros = np.ma.masked_where(dataFinalInt<=(1e-10), dataFinalInt)
        vmin = np.amin(dataWithoutZeros)
        vmax = np.amax(dataWithoutZeros)

        KSpecT = KSpecVsTimeAxis.pcolormesh(kgrid, tgrid, np.transpose(dataFinalInt), cmap='nipy_spectral', vmin=vmin, vmax=vmax, norm=LogNorm())
        #KSpecVsTimeFig.colors.Colormap('nipy_spectral')
        KSpecVsTimeFig.colorbar(KSpecT, ax=KSpecVsTimeAxis)
        KSpecVsTimeAxis.set_ylabel('$t$')
        KSpecVsTimeAxis.set_xlabel('$k_\parallel$')
        KSpecVsTimeAxis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        KSpecVsTimeAxis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]
        KSpecVsTimeFig.text(0.05,0.95,label)
        KSpecVsTimeAxis.set_title('Fourier spectrum vs time, Log-normalized $G^2$')
        self.addfig('k-spec vs time '+str(self.dict_id), KSpecVsTimeFig)               
        
    def createMSpecVsTime(self, t, mm, kpar, dataFinal):

        dataFinalInt=np.zeros((len(t),len(mm)))
        for i in range(len(mm)):
            for j in range(len(t)):
                dataFinalInt[j, i]=np.sum(dataFinal[j, i, :])
            
        tgrid, mgrid = np.meshgrid(t, mm)
        MSpecVsTimeFig = Figure()
        MSpecVsTimeAxis = MSpecVsTimeFig.add_subplot(111)

        dataWithoutZeros = np.ma.masked_where(dataFinalInt<=(1e-10), dataFinalInt)
        vmin = np.amin(dataWithoutZeros)
        vmax = np.amax(dataWithoutZeros)

        MSpecT=MSpecVsTimeAxis.pcolormesh(mgrid, tgrid, np.transpose(dataFinalInt), cmap='nipy_spectral', vmin=vmin, vmax=vmax, norm=LogNorm())
        MSpecVsTimeFig.colorbar(MSpecT, ax=MSpecVsTimeAxis)
        MSpecVsTimeAxis.set_ylabel('$t$')
        MSpecVsTimeAxis.set_xlabel('$m$')
        MSpecVsTimeAxis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        MSpecVsTimeAxis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]
        MSpecVsTimeFig.text(0.05,0.95,label)
        MSpecVsTimeAxis.set_title('Hermite spectrum vs time, Log-normalized $G^2$')
        self.addfig('m spec vs time '+str(self.dict_id), MSpecVsTimeFig)
        
    def create2DSpectra(self, mm, kpar, dataFinal, typeIdx):
        #Create 2d spectra, averaged over time
        
        mgrid, kpargrid=np.meshgrid(mm, kpar)
        Spec2DFig = Figure()            
        Spec2DAxis = Spec2DFig.add_subplot(111)
        
        dataWithoutZeros = np.ma.masked_where(dataFinal<=1e-16, dataFinal) 
        vmin = np.amin(dataWithoutZeros)
        vmax = np.amax(dataWithoutZeros)
      
        magSpec2D=Spec2DAxis.pcolormesh(kpargrid, mgrid, np.transpose(dataFinal), cmap='nipy_spectral', vmin=vmin, vmax=vmax, norm=LogNorm())
        Spec2DFig.colorbar(magSpec2D, ax=Spec2DAxis)
        Spec2DAxis.set_ylabel('$m$')
        Spec2DAxis.set_xlabel("$k_\parallel$") 
        Spec2DAxis.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        Spec2DAxis.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        
        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        Spec2DFig.text(0.05,.95,label)
        
        if typeIdx==0:
            Spec2DAxis.set_title('2D Log-Normalized Spectrum ($G^2$)')
            self.addfig('2D Spec (G^2)_'+self.averageType+self.dataCombineType+str(self.dict_id), Spec2DFig)   
        elif typeIdx==1:
            Spec2DAxis.set_title('Total 2D Log-Normalized Spectrum (Kinetic Energy)')
            self.addfig('2D Spec (Kinetic Energy)_'+self.averageType+self.dataCombineType+str(self.dict_id), Spec2DFig)
        else:
            Spec2DAxis.set_title('Total 2D Log-Normalized Spectrum (Magnetic Energy)')
            self.addfig('2D Spec (Magnetic Energy)_'+self.averageType+self.dataCombineType+str(self.dict_id), Spec2DFig)
            
    def createMSpecFixedK(self, mm, kpar, G2DataFinal):
        #Hermite spectrum at fixed kpar
        
        #get fixed kpar points
        fixedK=self.fixedKSpinBox.value()
        
        #Get data at fixed kpar
        index=np.where(kpar==fixedK) 
        G2DataFinal=G2DataFinal[:,(index[0])[0]]            
        
        #Name figure
        figName='Hermite spectrum (k='+str(fixedK)+')_'+self.averageType+self.dataCombineType+str(self.dict_id)
        
        #Create figure
        MSpecFixedKFig = Figure()            
        MSpecFixedKAxis = MSpecFixedKFig.add_subplot(111)
        MSpecFixedKAxis.set_yscale('log')
        MSpecFixedKAxis.set_xscale('log')
        
        mm = np.ma.masked_where(G2DataFinal<=0, mm)
        G2DataFinal = np.ma.masked_where(G2DataFinal<=0, G2DataFinal)
               
        MSpecFixedKAxis.scatter(mm[:], G2DataFinal, color='blue',s=30,edgecolor='none', label='$G^2$')  

        #estimate the slope of the spectrum
        slope, c=self.getSlope(np.log(mm[5:]), np.log(G2DataFinal[5:]))
        
        #add the slope to the slope dictionary
        self.slope_dict[figName]=slope  
        
        x = np.arange(mm[5], mm[-1], 1)
        y = (2.718**c)*x**(slope)
        MSpecFixedKAxis.plot(x,y, label='Fit')
        
        #Create slope label
        MSpecFixedKFig.text(0.80, 0.95, "p="+'%.5f' % (-1*slope), label="plabel")        
        MSpecFixedKAxis.set_title('$m$ Spectrum for Fixed $k_\parallel$='+str(fixedK))
        MSpecFixedKAxis.set_xlabel('$m$')
        MSpecFixedKAxis.set_ylabel('$G^2$')  

        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        MSpecFixedKFig.text(0.05,.95,label)    
                     
        self.addfig(figName, MSpecFixedKFig)
   
    def createKSpecFixedM(self, mm, kpar, G2DataFinal):
        #1D kpar spectra at fixed m
        #get fixed m points
        fixedM=self.fixedMSpinBox.value()
        
        #Get data at fixed m
        index=np.where(mm==fixedM)
        G2DataFinal=G2DataFinal[(index[0])[0], :]
        
        #name the figure
        figName='k-spectrum (m='+str(fixedM)+')_'+self.averageType+self.dataCombineType+str(self.dict_id)        
               
        #Create figure
        KSpecFixedMFig  = Figure()            
        KSpecFixedMAxis = KSpecFixedMFig.add_subplot(111)
          
        #Mask values where G**2 is 0
        kpar = np.ma.masked_where(G2DataFinal<=0, kpar)        
        G2DataFinal = np.ma.masked_where(G2DataFinal<=0, G2DataFinal)

        kpar[0]=0.5
        
        KSpecFixedMAxis.get_xaxis()
        KSpecFixedMAxis.get_yaxis()          
        KSpecFixedMAxis.set_yscale('log')
        KSpecFixedMAxis.set_xscale('log')

        KSpecFixedMAxis.scatter(kpar[:], G2DataFinal, color='blue',s=30,edgecolor='none', label='$G^2$')  
        KSpecFixedMAxis.set_title('$k_\parallel$ Spectrum for fixed $m$='+str(fixedM))
        KSpecFixedMAxis.set_xlabel('$k_\parallel$')
        KSpecFixedMAxis.set_ylabel('$G^2$')
        
        #Add run label        
        fileName=self.fileListDisplay.currentItem().text()
        label='Run: '+fileName[0:fileName.indexOf('.')]        
        KSpecFixedMFig.text(0.05,.95,label)        
        
        self.addfig(figName, KSpecFixedMFig)   
   
    def getSlope(self, x, y):
        slope, c=np.polyfit(x,y, deg=1)
        return (slope, c)
   
    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)

    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def addfig(self, name, fig):
        self.fig_dict[QtCore.QString(name)] = fig        
        self.figList.addItem(name)
        self.dict_id+=1

    def changefig(self, item):
        text = item.text()
        self.rmmpl()
        self.addmpl(self.fig_dict[text])

if __name__ == '__main__':
    import sys
    import os
    from PyQt4 import QtGui, QtCore
    import numpy as np
    import numpy.ma as ma

    app=QtGui.QApplication.instance() # checks if QApplication already exists
    if not app: # create QApplication if it doesnt exist
         app = QtGui.QApplication(sys.argv)
         
    #take command line input if there is any
    files=QtCore.QStringList()
    if len(sys.argv)>1:
        for i in range(1,len(sys.argv)):
            file=sys.argv[i] 
            if(not os.path.isfile(file)):
                print "Input file does not exist"
                sys.exit()
            else:
                files.append(QtCore.QString(file))      
    main = Main(files)
    main.show()
    sys.exit(app.exec_())
