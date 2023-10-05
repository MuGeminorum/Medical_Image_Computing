from __main__ import vtk, qt, ctk, slicer
from vtk.util import numpy_support
import numpy as np

def offset(arrays, output = None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype    
    n = np.prod([x.size for x in arrays])
    if output is None:
        output = np.zeros([n, len(arrays)], dtype = dtype)    
    m = n / arrays[0].size
    output[:,0] = np.repeat(arrays[0], m)    
    if arrays[1:]:
        offset(arrays[1:], output = output[0:m, 1:])
        for j in range(1, arrays[0].size):
            output[j*m:(j+1)*m, 1:] = output[0:m, 1:]    
    return output

# find the coordinates of new voxels
def find_new_voxels(sx, sy, sz, iteration):
  '''
   Function Declaration
   - input: [sx, sy, sz]: the spatial location of the seed
            iteration: the current iteration number, which also can be used to represent the new searching length (radius).

   - output: new_voxel_coordinates: a numpy array which contains the coordinates of the voxels to be examined during this iteration

   - tips: 1.new_voxel_coordinates is initialized as an empty list by tutors, and it will be converted to numpy
             array at the end of this function 
           2.during each iteration, we will look at:
             (a) the voxels at sx-iteration and sx+iteration in x-direction,from sy-iteration to sy+iteration in y-direction 
                 and from sz-iteration to sz+iteration in z-direction; 
             (b) the voxels at sy-iteration and sy+iteration in y-direction,from sx-iteration to sx+iteration in x-direction 
                 and from sz-iteration to sz+iteration in z-direction; 
             (c) the voxels at sz-iteration and sz+iteration in z-direction,from sx-iteration to sx+iteration in x-direction 
                 and from sy-iteration to sy+iteration in y-direction; 
           3. You need to retrieve all these voxels locations (except for those voxels in the corners, which you only need to
              add them once) and put them into new_voxel_coordinates

  '''
  new_voxel_coordinates = []

  #########################
  #Add your code below    #
  #########################
  new_voxel_coordinates_yz = offset((np.array([sx-iteration, sx+iteration]), np.arange(sy-iteration, sy+iteration+1), np.arange(sz-iteration, sz+iteration+1)))
  new_voxel_coordinates_xz = offset((np.arange(sx-iteration+1, sx+iteration), np.array([sy-iteration, sy+iteration]), np.arange(sz-iteration, sz+iteration+1)))
  new_voxel_coordinates_xy = offset((np.arange(sx-iteration+1, sx+iteration), np.arange(sy-iteration+1, sy+iteration), np.array([sz-iteration, sz+iteration])))
  new_voxel_coordinates = np.concatenate((new_voxel_coordinates_yz, np.concatenate((new_voxel_coordinates_xz, new_voxel_coordinates_xy))))
  #########################
  #Add your code above    #
  #########################
  new_voxel_coordinates = np.asarray(new_voxel_coordinates)
  print('[Iter {}] the shape of new_voxel_coordinates in this iteration: {}'.format(iteration, new_voxel_coordinates.shape))

  return new_voxel_coordinates


#
# RegionGrowing
#
def grow_from_seed(inputVolumeData, size, seed_location, global_intensity_diff, local_intensity_diff, outputROIData):
  '''
     Function Declaration
     - input: inputVolumeData: a numpy array representing the raw image input.
              size: a list contains the size of the inputVolumeData, already loaded as ['dx'(length) 
                    , 'dy'(width) and 'dz'(height)]
              seed_location: a list contains the location of the seed, already loaded as ['sx', 'sy' and 'sz'] 
              global_intensity_diff: a list contains the global intensity range threshold, already
                                    loaded as ['ROI_min'(minimum intensity threshold) and 'ROI_max'
                                    (maximum intensity threshold)]
              local_intensity_diff: the local intensity threshold used to calculate local intensity range

     - output: outputROIData: a numpy array(has the same size of inputVolumeData) representing the binary 
               segmentation mask, where 0 represents background(not tumor) and 1 represents the tumor

     - tips: 1.For each iteration, you need to:
                * find the voxels to be examined in this iteration by using 'find_new_voxels'
                * GlobalChecking: compare the intensity value of each voxel to be examined in this iteration 
                  with global intensity range
                * LocalChecking: compare the intensity value of each voxel to be examined in this iteration 
                  with its corresponding local intensity range
                * consider the stopping criterion, i.e, if the pixels to be examined reach the 
                  boundary of the image 
             2.If you have no idea where to start, use a while True loop and set the break when stopping criterion is met
               (Try to run the first iteration by adding a break in this function to stop the algorithm to check if the result is expected)
             
  '''
  dx, dy, dz = size
  sx, sy, sz = seed_location
  ROI_min, ROI_max = global_intensity_diff
  local_intensity_diff = local_intensity_diff

  iteration = 0
  # the local searching radius
  radius = 1

  # Stopping criterion: reach the boundary of the image
  # GlobalChecking:  whether the voxel value is in the global intensity range
  # LocalChecking: whether the voxel value is in its corresponding local intensity range
  #########################
  #Add your code below    #
  #########################
  # Hint: you are encouraged to use "print()" to check these vairables status below, to help you gain a better understanding on this algorithm
  while 1 > 0:

    iteration += 1
    searching_extend = np.array([iteration+radius-sx, sx+iteration+radius+1-dx, iteration+radius-sy, sy+iteration+radius+1-dy, iteration+radius-sz, sz+iteration+radius+1-dz])
    
    if (searching_extend >= 0).any():
        break

    new_voxel_coords = find_new_voxels(sx, sy, sz, iteration)
    new_voxel_values = inputVolumeData[new_voxel_coords[:, 0], new_voxel_coords[:, 1], new_voxel_coords[:, 2]]
    glb_voxel_indicators = np.where(np.logical_and(new_voxel_values < ROI_max, new_voxel_values > ROI_min))

    if not glb_voxel_indicators:
        break

    else:
        for i in glb_voxel_indicators[0]:
            nx, ny, nz = new_voxel_coords[i, :]
            patch_boolen = outputROIData[nx-1:nx+2, ny-1:ny+2, nz-1:nz+2]
                
            if patch_boolen.sum() > 1:
                local_value = inputVolumeData[nx, ny, nz]
                patch_values = inputVolumeData[nx-1:nx+2, ny-1:ny+2, nz-1:nz+2]
                boolean_values = patch_values[:]*patch_boolen[:]
                existing_values = boolean_values[np.where(boolean_values > 0)]
                local_min = existing_values.min()-local_intensity_diff
                local_max = existing_values.max()+local_intensity_diff

            if local_value > local_min and local_value < local_max:
                outputROIData[nx, ny, nz] = 1
  #########################
  #Add your code above    #
  #########################
  return outputROIData





############################################## 
# Template Code                              #
# * Setup the connection with 3D Slicer      #
# * Setup the button in Slicer               #
# * [!] Please do not modify any line below  #
##############################################
#
# MedicalImageSegmentation
#
class MedicalImageSegmentation:
  def __init__(self, parent):
    parent.title = "Task B - MIS"
    parent.categories = ["Assignment"]
    parent.dependencies = []
    parent.contributors = [ """
    Sidong Liu (USYD) 
    Siqi Liu (Simense)
    Zihao Tang (USYD)
    Chaoyi Zhang (USYD)
    """]
    parent.helpText = """
    Task B. Medical Image Segmentation
    """
    parent.acknowledgementText = """
    This python program shows a simple implementation of the 3D region growing based segmentation for 
    the Assignment of COMP5424. 
    """ 
    self.parent = parent

#
# The main widget
#
class MedicalImageSegmentationWidget:
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
    self.testReloadFrame = ctk.ctkCollapsibleButton()
    self.testReloadFrame.objectName = 'ReloadFrame'
    self.testReloadFrame.setLayout(qt.QHBoxLayout())
    self.testReloadFrame.setText("Reload the Module")
    self.layout.addWidget(self.testReloadFrame)
    self.testReloadFrame.collapsed = False
 
    # Add a reload button for debug
    reloadButton = qt.QPushButton("Reload")
    reloadButton.toolTip = "Reload this Module"
    reloadButton.name = "MedicalImageSegmentation Reload"
    reloadButton.connect('clicked()', self.onReload)
    self.reloadButton = reloadButton
    self.testReloadFrame.layout().addWidget(self.reloadButton)
    
    # Collapsible button
    self.laplaceCollapsibleButton = ctk.ctkCollapsibleButton()
    self.laplaceCollapsibleButton.text = "3D Region Grow Inputs"
    self.layout.addWidget(self.laplaceCollapsibleButton)

    # Layout within the laplace collapsible button
    self.segmentationFormLayout = qt.QFormLayout(self.laplaceCollapsibleButton)
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addEnabled = True
    self.inputSelector.removeEnabled = True
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationFormLayout.addRow("Input Volume: ", self.inputSelector)
    self.seedingSelector = slicer.qMRMLNodeComboBox()
    self.seedingSelector.nodeTypes = ( ("vtkMRMLLabelMapVolumeNode"), "" )
    self.seedingSelector.addEnabled = True
    self.seedingSelector.removeEnabled = True
    self.seedingSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationFormLayout.addRow("Seeding ROI: ", self.seedingSelector)

    # Change the parameters
    updateParameterCollapsibleButtion      = ctk.ctkCollapsibleButton()
    updateParameterCollapsibleButtion.text = "Selection Criteria"
    self.layout.addWidget(updateParameterCollapsibleButtion)
    updateParameterFormLayout              = qt.QFormLayout(updateParameterCollapsibleButtion)


    chooseGlobalFrame, chooseGlobalSlider, chooseGlobalSliderSpinBox = numericInputFrame(self.parent, \
                                                              "Maximum Global Intensity Difference:   ", \
                                                              "Determin the global range of intensity values", \
                                                              1, 50, 1, 0)
    updateParameterFormLayout.addWidget(chooseGlobalFrame)

    chooseLocalFrame, chooseLocalSlider, chooseLocalSliderSpinBox = numericInputFrame(self.parent, \
                                                              "Maximum Local Intensity Difference:     ", \
                                                              "Determine the local range of of intensity values", 0, 50, 1, 0)
    updateParameterFormLayout.addWidget(chooseLocalFrame)

    # Apply button
    applyButton = qt.QPushButton("Apply")
    applyButton.toolTip = "Run the Image Segmentation."
    self.layout.addWidget(applyButton)
    applyButton.connect('clicked(bool)', self.onApply)
    self.applyButton = applyButton
    
    class state(object):
      maxGlobalDiff    = 20
      maxLocalDiff     = 20

    scopeLocals    = locals()

    def connect(obj, evt, cmd):
      def callback(*args):
        currentLocals = scopeLocals.copy()
        currentLocals.update({'args':args})
        exec(cmd, globals(), currentLocals)
        updateGUI()
      obj.connect(evt, callback)

    def updateGUI():
      chooseGlobalSlider.value          = state.maxGlobalDiff
      chooseGlobalSliderSpinBox.value   = state.maxGlobalDiff
      chooseLocalSlider.value           = state.maxLocalDiff
      chooseLocalSliderSpinBox.value    = state.maxLocalDiff
      
    connect(chooseGlobalSlider, 'valueChanged(double)', 'state.maxGlobalDiff = args[0]')
    connect(chooseGlobalSliderSpinBox, 'valueChanged(double)', 'state.maxGlobalDiff = args[0]')
    connect(chooseLocalSlider, 'valueChanged(double)', 'state.maxLocalDiff = args[0]')
    connect(chooseLocalSliderSpinBox, 'valueChanged(double)', 'state.maxLocalDiff = args[0]')

    updateGUI()
    self.updateGUI  = updateGUI
    self.state      = state
    self.layout.addStretch(1)


  # When the apply button is clicked
  def onApply(self):
    # Read in the input volume
    inputVolume     = self.inputSelector.currentNode()
    inputVolumeData = slicer.util.array(inputVolume.GetID())
    
    # Read in the seeding ROI
    seedingROI      = self.seedingSelector.currentNode()
    seedingROIData  = slicer.util.array(seedingROI.GetID())
    
    # Copy image node, create a new volume node
    outputROI_name  = seedingROI.GetName() + '_grow'
    outputROI       = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, seedingROI, outputROI_name)
    outputROIData   = slicer.util.array(outputROI.GetID())

    # print(outputROIData)
    
    # Get the mean of the seeding ROI
    seedingROI_coords   = np.where(seedingROIData > 0)
    print(seedingROI_coords)
    seedingROI_values   = inputVolumeData[seedingROI_coords]
    
    # # the location of the seeding voxel
    sx = seedingROI_coords[0][seedingROI_values.argmax()]
    sy = seedingROI_coords[1][seedingROI_values.argmax()]
    sz = seedingROI_coords[2][seedingROI_values.argmax()]

    # The global parameter is used to select the voxels within a range  
    ROI_min = seedingROI_values.min() - self.state.maxGlobalDiff 
    ROI_max = seedingROI_values.max() + self.state.maxGlobalDiff

    # Dimension of the input volume 
    dx, dy, dz = inputVolumeData.shape
            
    # iteration = 0
    # # the local searching radius
    # radius = 1

    outputROIData = grow_from_seed(inputVolumeData, [dx, dy, dz], [sx, sy, sz], [ROI_min, ROI_max], self.state.maxLocalDiff, outputROIData)

    outputROI.GetImageData().Modified()
    
    # make the output volume appear in all the slice views
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetReferenceActiveLabelVolumeID(outputROI.GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection(0)
    
  # 
  # Supporting Functions
  # 
  # Reload the Module
  def onReload(self, moduleName = "MedicalImageSegmentation"):
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


# Numeric parameter input
def numericInputFrame(parent, label, tooltip, minimum, maximum, step, decimals):
  inputFrame              = qt.QFrame(parent)
  inputFrame.setLayout(qt.QHBoxLayout())
  inputLabel              = qt.QLabel(label, inputFrame)
  inputLabel.setToolTip(tooltip)
  inputFrame.layout().addWidget(inputLabel)
  inputSpinBox            = qt.QDoubleSpinBox(inputFrame)
  inputSpinBox.setToolTip(tooltip)
  inputSpinBox.minimum    = minimum
  inputSpinBox.maximum    = maximum
  inputSpinBox.singleStep = step
  inputSpinBox.decimals   = decimals
  inputFrame.layout().addWidget(inputSpinBox)
  inputSlider             = ctk.ctkDoubleSlider(inputFrame)
  inputSlider.minimum     = minimum
  inputSlider.maximum     = maximum
  inputSlider.orientation = 1
  inputSlider.singleStep  = step
  inputSlider.setToolTip(tooltip)
  inputFrame.layout().addWidget(inputSlider)
  return inputFrame, inputSlider, inputSpinBox
