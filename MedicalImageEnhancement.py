from __main__ import vtk, qt, ctk, slicer
from vtk.util import numpy_support
import numpy as np

def convolution3D(inputData, filter_name):
  '''
    Function Declaration
    - Function Name: convolution3D
    - Function Input: 
            - inputData: the 3D input tensor to be convolved
            - filter_name: the filter to be used
                           it can be one of (Smoothing, Sharpening, Edge)
    - Function Output: 
            - outputData: convolved 3D medical image
    - Function Usage:
            - Perform 3D convolution specified by "filter_name", on 3D MRI data "inputData"
            - Save and return the output result "outputData" 
    - Tips:
            - This task can be divided into 2 stages, namely, Kernel Definition and ConvolutionIteration
            - We only focus on 3x3x3 kernel in this assignment.
  '''
  dx, dy, dz = inputData.shape 
  outputData = np.zeros((dx, dy, dz))
  ############### Add your code below ######################
  # Stage 1: Kernel Definition
  # Define your 3x3x3 kernel.
  if filter_name== "Smoothing":
    print("INFO: Applying a smoothing filter ...")
    # Tutors have already implemented this filter for you.
    kernel = np.ones([3, 3, 3])
    kernel = kernel / kernel.sum()

    from datetime import datetime
    dt = datetime.now()

    for i in range(dx-2):
    	for j in range(dy-2):
    		for k in range(dz-2):
    			outputData[i+1, j+1, k+1] = np.sum(inputData[i:i+3, j:j+3, k:k+3]*kernel)
    
    print("Time cost: " + str((datetime.now() - dt).seconds) + "s")		

  elif filter_name == "Sharpening":
    print("INFO: Applying a sharpening filter ...")
    # You need to implement 3x3x3 sharpening filter by yourself
    kernel = -np.ones([3, 3, 3])
    kernel[0][1][1] = 9
    kernel[1][1][1] = 9
    kernel[2][1][1] = 9

    from datetime import datetime
    dt = datetime.now()

    for i in range(dx-2):
    	for j in range(dy-2):
    		for k in range(dz-2):
    			outputData[i+1, j+1, k+1] = np.sum(inputData[i:i+3, j:j+3, k:k+3]*kernel)
    
    print("Time cost: " + str((datetime.now() - dt).seconds) + "s")		

  elif filter_name == "Edge":
    print("INFO: Applying an edge detection filter ...")
    # You need to implement 3x3x3 edge detection filter by yourself
    kernel = np.zeros([3, 3, 3])
    kernel[0] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel[1] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel[2] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    from datetime import datetime
    dt = datetime.now()	

    for i in range(dx-2):
    	for j in range(dy-2):
    		for k in range(dz-2):
    			outputData[i+1, j+1, k+1] = np.sum(inputData[i:i+3, j:j+3, k:k+3]*kernel)
    
    print("Time cost: " + str((datetime.now() - dt).seconds) + "s")	

  # Stage 2: ConvolutionIteration
  # Iterate all positions in the image with your kernel.
  # Some variable you may use: dx, dy, dz, inputData, kernel, outputData
  # You need to implement your convolution operation here.
  # ...
  # ...
  # ...
  ############### Add your code above ######################
  return outputData



############################################## 
# Template Code                              #
# * Setup the connection with 3D Slicer      #
# * Setup the button in Slicer               #
# * [!] Please do not modify any line below  #
##############################################
#
# MedicalImageEnhancement
#
class MedicalImageEnhancement:
  def __init__(self, parent):
    parent.title = "Task A - MIE"
    parent.categories = ["Assignment"]
    parent.dependencies = []
    parent.contributors = [ """
    Sidong Liu (USYD) 
    Siqi Liu (Simense)
    Chaoyi Zhang (USYD)
    Zihao Tang (USYD)
    """]
    parent.helpText = """
    Task A. Medical Image Enhancement
    """
    parent.acknowledgementText = """
    This python program shows a simple implementation of the 3D convolution filtering for 
    the Assignment of COMP5424. 
    """ 
    self.parent = parent

#
# The main widget
#

class MedicalImageEnhancementWidget:
  def __init__(self, parent = None):
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()

  #  Setup the layout
  def setup(self):
    # Collapsible button
    self.laplaceCollapsibleButton = ctk.ctkCollapsibleButton()
    self.laplaceCollapsibleButton.text = "Image Filter"
    self.layout.addWidget(self.laplaceCollapsibleButton)

    # Layout within the laplace collapsible button
    self.filterFormLayout = qt.QFormLayout(self.laplaceCollapsibleButton)

    # the volume selectors
    self.inputFrame = qt.QFrame(self.laplaceCollapsibleButton)
    self.inputFrame.setLayout(qt.QHBoxLayout())
    self.filterFormLayout.addWidget(self.inputFrame)
    self.inputSelector = qt.QLabel("Input Volume: ", self.inputFrame)
    self.inputFrame.layout().addWidget(self.inputSelector)
    self.inputSelector = slicer.qMRMLNodeComboBox(self.inputFrame)
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputFrame.layout().addWidget(self.inputSelector)

    # Add a reload button for debug
    reloadButton = qt.QPushButton("Reload")
    reloadButton.toolTip = "Reload this Module"
    reloadButton.name = "MedicalImageEnhancement Reload"
    reloadButton.connect('clicked()', self.onReload)
    self.reloadButton = reloadButton
    self.filterFormLayout.addWidget(self.reloadButton)

    # Add a clear screen button for debug
    clearScreenButton = qt.QPushButton("Clear Screen")
    clearScreenButton.toolTip = "Clear Python Interactor Screen"
    clearScreenButton.name = "ClearScreen"
    clearScreenButton.connect('clicked()', self.onClearScreen)
    self.clearScreenButton = clearScreenButton
    self.filterFormLayout.addWidget(self.clearScreenButton)

    # Choose the filter
    self.filter = "Smoothing"

    changeFilterFrame = qt.QFrame(self.parent)
    changeFilterFrame.setLayout(qt.QVBoxLayout())
    self.filterFormLayout.addWidget(changeFilterFrame)
    self.changeFilterFrame = changeFilterFrame

    chooseSmooth = qt.QRadioButton("Smoothing")
    chooseSmooth.setChecked(True)
    chooseSmooth.connect('clicked()', self.chooseSmooth)
    self.filterFormLayout.addWidget(chooseSmooth)
    self.chooseSmooth = chooseSmooth

    chooseSharpen = qt.QRadioButton("Sharpening")
    chooseSharpen.connect('clicked()', self.chooseSharpen)
    self.filterFormLayout.addWidget(chooseSharpen)
    self.chooseSharpen = chooseSharpen

    chooseEdge = qt.QRadioButton("Edge Detection")
    chooseEdge.connect('clicked()', self.chooseEdge)
    self.filterFormLayout.addWidget(chooseEdge)
    self.chooseEdge = chooseEdge

    # Apply button
    filterButton = qt.QPushButton("Apply")
    filterButton.toolTip = "Run the Image Filtering."
    self.filterFormLayout.addWidget(filterButton)
    filterButton.connect('clicked(bool)', self.onApply)
    self.filterButton = filterButton

    # Add vertical spacer
    self.layout.addStretch(1)

  # Choose what filter to use 
  def chooseSharpen(self):
    self.filter = "Sharpening"

  def chooseSmooth(self):
    self.filter = "Smoothing"

  def chooseEdge(self):
    self.filter = "Edge"


  # When the apply button is clicked
  def onApply(self):
    # Read in the image node
    inputVolume = self.inputSelector.currentNode()
    # Extract the array
    inputVolumeData = slicer.util.array(inputVolume.GetID())
    # Name the output volume
    outputVolume_name = inputVolume.GetName() + '_filtered'
    # Copy image node, create a new volume node
    volumesLogic = slicer.modules.volumes.logic()
    outputVolume = volumesLogic.CloneVolume(slicer.mrmlScene, inputVolume, outputVolume_name)
    # Find the array that is associated with the label map
    outputVolumeData = slicer.util.array(outputVolume.GetID())
    # the dimensions of the output volume
    dx, dy, dz = outputVolumeData.shape 

    outputVolumeData[:] = convolution3D(inputVolumeData, self.filter)
    outputVolume.GetImageData().Modified()
    
    # make the output volume appear in all the slice views
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetReferenceActiveVolumeID(outputVolume.GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection(0)
  # 
  # Supporting Functions
  # 

  # Reload the Module
  def onReload(self, moduleName = "MedicalImageEnhancement"):
    import imp, sys, os, slicer

    widgetName = moduleName + "Widget"
    fPath = eval('slicer.modules.%s.path' % moduleName.lower())
    p = os.path.dirname(fPath)
    if not sys.path.__contains__(p):
      sys.path.insert(0,p)
    fp = open(fPath, "r")
    globals()[moduleName] = imp.load_module(
        moduleName, fp, fPath, ('.py', 'r', imp.PY_SOURCE))
    fp.close()

    print("the module name to be reloaded,", moduleName)
    # find the Button with a name 'moduleName Reolad', then find its parent (e.g., a collasp button) and grand parent (moduleNameWidget)
    parent = slicer.util.findChildren(name = '%s Reload' % moduleName)[0].parent().parent()
    for child in parent.children():
      try:
        child.hide()
      except AttributeError:
        pass

    item = parent.layout().itemAt(0)
    while item:
      parent.layout().removeItem(item)
      item = parent.layout().itemAt(0)

    globals()[widgetName.lower()] = eval('globals()["%s"].%s(parent)' % (moduleName, widgetName))
    globals()[widgetName.lower()].setup()


  # Clear the Python Interacter Screen 
  def onClearScreen(self):
    print("\n" * 50)


