import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# colors borrowed from Caleb Robinson's landcover mapping repo.

NLCD_NODATA = 0

# Copied from https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend
NLCD_CLASS_DEFINITIONS = {
    0: ("No Data", "No data value"),
    11: ("Open Water", "Areas of open water, generally with less than 25% cover of vegetation or soil."),
    12: ("Ice/Snow", "Areas characterized by a perennial cover of ice and/or snow, generally greater than 25% of total cover."),
    21: ("Developed Open Space", "Areas with a mixture of some constructed materials, but mostly vegetation in the form of lawn grasses. Impervious surfaces account for less than 20% of total cover. These areas most commonly include large-lot single-family housing units, parks, golf courses, and vegetation planted in developed settings for recreation, erosion control, or aesthetic purposes."),
    22: ("Developed Low Intensity", "Areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 20% to 49% percent of total cover. These areas most commonly include single-family housing units."),
    23: ("Developed Medium Intensity", "Areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 50% to 79% of the total cover. These areas most commonly include single-family housing units."),
    24: ("Developed High Intensity", "Highly developed areas where people reside or work in high numbers. Examples include apartment complexes, row houses and commercial/industrial. Impervious surfaces account for 80% to 100% of the total cover."),
    31: ("Barren Land (Rock/Sand/Clay)", "Areas of bedrock, desert pavement, scarps, talus, slides, volcanic material, glacial debris, sand dunes, strip mines, gravel pits and other accumulations of earthen material. Generally, vegetation accounts for less than 15% of total cover."),
    41: ("Deciduous Forest", "Areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species shed foliage simultaneously in response to seasonal change."),
    42: ("Evergreen Forest", "Areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species maintain their leaves all year. Canopy is never without green foliage."),
    43: ("Mixed Forest", "Areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. Neither deciduous nor evergreen species are greater than 75% of total tree cover."),
    52: ("Shrub/Scrub", "Areas dominated by shrubs; less than 5 meters tall with shrub canopy typically greater than 20% of total vegetation. This class includes true shrubs, young trees in an early successional stage or trees stunted from environmental conditions."),
    71: ("Grassland/Herbaceous", "Areas dominated by gramanoid or herbaceous vegetation, generally greater than 80% of total vegetation. These areas are not subject to intensive management such as tilling, but can be utilized for grazing."),
    81: ("Pasture/Hay", "Areas of grasses, legumes, or grass-legume mixtures planted for livestock grazing or the production of seed or hay crops, typically on a perennial cycle. Pasture/hay vegetation accounts for greater than 20% of total vegetation."),
    82: ("Cultivated Crops", "Areas used for the production of annual crops, such as corn, soybeans, vegetables, tobacco, and cotton, and also perennial woody crops such as orchards and vineyards. Crop vegetation accounts for greater than 20% of total vegetation. This class also includes all land being actively tilled."),
    90: ("Woody Wetlands", "Areas where forest or shrubland vegetation accounts for greater than 20% of vegetative cover and the soil or substrate is periodically saturated with or covered with water."),
    95: ("Emergent Herbaceous Wetlands", "Areas where perennial herbaceous vegetation accounts for greater than 80% of vegetative cover and the soil or substrate is periodically saturated with or covered with water.")
}

# Copied from the emebedded color table in the NLCD data files
NLCD_CLASS_COLORS = {
    0:  (0, 0, 0, 0),
    11: (70, 107, 159, 255),
    12: (209, 222, 248, 255),
    21: (222, 197, 197, 255),
    22: (217, 146, 130, 255),
    23: (235, 0, 0, 255),
    24: (171, 0, 0, 255),
    31: (179, 172, 159, 255),
    41: (104, 171, 95, 255),
    42: (28, 95, 44, 255),
    43: (181, 197, 143, 255),
    52: (204, 184, 121, 255),
    71: (223, 223, 194, 255),
    81: (220, 217, 57, 255),
    82: (171, 108, 40, 255),
    90: (184, 217, 235, 255),
    95: (108, 159, 184, 255)
}


ENVIROATLAS_NODATA = 0

ENVIROATLAS_CLASS_DEFINITIONS = {
    0: ("Unclassified", r"No data value"),
    10: ("Water", "The water class includes all surface waters: streams, rivers, canals, ponds, reservoirs, lakes, bays, estuaries, and coastal waters. For cases of ephemeral changes in water level and extent such as tidelands and some lakes, the waterline at the time of photo acquisition is used to define the extent of the water feature."),
    20: ("Impervious Surface", "An impervious surface is a landscape feature that prevents or substantially limits rainfall from infiltrating into the soil, including: paved roads, parking lots, driveways, sidewalks, roofs, swimming pools, patios, painted surfaces, wooden structures, and most asphalt and concrete surfaces. Many dirt and gravel roads, and railways, are functionally impervious or semi-impervious and are included in the impervious surface class. Most impervious surfaces are anthropogenic. When trees overhang streets and other impervious surfaces, those pixels are assigned to the Tree class rather than the underlying Impervious class. This assignment reflects the EnviroAtlas emphasis on ecosystem services, and the importance of street trees in urban areas."),
    30: ("Soil and Barren", "The Soil and Barren class includes soil, bare rock, mud, clay and sand. This class includes bare fields, construction sites, quarries, gravel pits, mine lands, golf sand traps, stream and river sand bars, beaches and other bare soil and gravel surfaces. Soil and Barren includes natural areas with widely spaced or no vegetation cover."),
    40: ("Trees and Forest", "The Trees and Forest class includes trees of any kind, from a single individual to continuous canopy forest. If a vegetation object casts a shadow longer than a few meters, it is usually classified as a tree. Large shrubs fall in this class."),
    52: ("Shrubs", "Shrubs are generally shorter than trees and bear multiple woody stems. Shrubs are typically recognized in air photos by context (e.g., landscaping vegetation), the mottled texture of the canopy, and short shadows.  In the arid west, small woody vegetation occurring within a soil matrix are classified as shrubs.  In the humid, temperate eastern U.S., shrubs are typically not broken out as a separate class in the EnviroAtlas unless high quality LiDAR is available. If Shrubs are broken out in the classification, it is by height: <= 2m. Otherwise, shrubs typically are assigned to Tree class."),
    70: ("Grass and Herbaceous", "The Grass and Herbaceous class includes the gramminoids, forbs and herbs lacking persistent woody stems. Grass includes residential lawns, golf courses, roadway medians and verges, park lands, transmission line, natural gas corridors, recently clear cut areas, pasture, grasslands, prairie grass, and emergent wetlands vegetation. Small shrubs fall into this category."),
    80: ("Agriculture", "The Agriculture class includes herbaceous vegetation planted or being managed for the production of food, feed, or fiber. Agriculture includes cultivated row crops and fallow fields that are being actively tilled. Agriculture is typically a relatively rare class in urban areas, but may occur with greater frequency in exurban regions away from the urban core. The US Census Urban Areas can be large in aerial extent, and may encompass significant amounts of agricultural land. The Atlas treats agricultural lands primarily in a land cover sense as Grass-Herbaceous, Trees, Shrubs or Soil, and secondarily as Agriculture land use."),
    82: ("Orchards", "Orchards are trees planted or maintained for the production of fruits and timber."),
    91: ("Woody Wetlands", "Woody Wetlands are wetlands dominated by Tree and Forest species. Typically these are identified using ancillary GIS layers (e.g., National Wetlands Inventory)."),
    92: ("Emergent Wetlands", "Emergent Wetlands are wetlands dominated by Grass and Herbaceous species. Typically these are identified using ancillary GIS layers (e.g., National Wetlands Inventory).")
}

ENVIROATLAS_CLASS_COLORS_orig = {
    0: (0, 0, 0, 0), #
    10: (0, 197, 255, 255), # from CC Water
    20: (156, 156, 156, 255), # from CC Impervious
    30: (255, 170, 0, 255), # from CC Barren
    40: (38, 115, 0, 255), # from CC Tree Canopy
    #52: (76, 230, 0, 255), # from CC Shrubland
    52: (204, 184, 121, 255), # from NLCD shrub
    70: (163, 255, 115, 255), # from CC Low Vegetation
    80: (220, 217, 57, 255), # from NLCD Pasture/Hay color
    82: (171, 108, 40, 255), # from NLCD Cultivated Crops
    91: (184, 217, 235, 255), # from NLCD Woody Wetlands
    92: (108, 159, 184, 255) # from NLCD Emergent Herbaceous Wetlands
}

ENVIROATLAS_CLASS_COLORS = {
    0: (255, 255, 255, 255), #
    1: (0, 197, 255, 255), # from CC Water
    2: (156, 156, 156, 255), # from CC Impervious
    3: (255, 170, 0, 255), # from CC Barren
    4: (38, 115, 0, 255), # from CC Tree Canopy
    5: (204, 184, 121, 255), # from NLCD shrub
    6: (163, 255, 115, 255), # from CC Low Vegetation
    7: (220, 217, 57, 255), # from NLCD Pasture/Hay color
    8: (171, 108, 40, 255), # from NLCD Cultivated Crops
    9: (184, 217, 235, 255), # from NLCD Woody Wetlands
    10: (108, 159, 184, 255), # from NLCD Emergent Herbaceous Wetlands
    11: (0,0,0,0), # extra for black
    12: (70, 100, 159, 255), # extra for dark blue
}

CHESAPEAKE_5_CLASS_COLORS = {
      #  -1: (0, 0, 0, 0),
        0: (0, 197, 255, 255),
        1: (38, 115, 0, 255),
        2: (163, 255, 115, 255),
        3: (255, 170, 0, 255),
        4: (156, 156, 156, 255),
        }

CHESAPEAKE_4_CLASS_COLORS = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (156, 156, 156, 255),
        }

CHESAPEAKE_4_NO_ZEROS_CLASS_COLORS = {
        0: (0, 197, 255, 255),
        1: (38, 115, 0, 255),
        2: (163, 255, 115, 255),
        3: (156, 156, 156, 255),
        }
    
    
CHESAPEAKE_7_CLASS_COLORS = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (255, 170, 0, 255),
        5: (156, 156, 156, 255),
        6: (0, 0, 0, 255),
    }
CHESAPEAKE_CLASS_COLORS_with_extra = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (255, 170, 0, 255),
        5: (156, 156, 156, 255),
        6: (0, 0, 0, 255),
        7: (197, 0, 255, 255),
        8: (0, 0, 0, 0),
        9: (0, 0, 0, 0),
        10: (0, 0, 0, 0),
        11: (0, 0, 0, 0),
        12: (0, 0, 0, 0),
        13: (0, 0, 0, 0),
        14: (0, 0, 0, 0),
        15: (0, 0, 0, 0),
    }

CHESAPEAKE_5_CLASS_DEFINITIONS = {
    -1: ("Unclassified", r"No data value"),
    0: ("Water", "Water"),
    1: ("Tree/Forest", "Tree/Forest"),
    2: ("Low vegetaion/field", "Low vegetaion/field"),
    3: ("Soil and Barren", "Soil and Barren"),
    4: ("Impervious both", "Impervious both")
}

CHESAPEAKE_4_CLASS_DEFINITIONS = {
    0: ("Unclassified", r"No data value"),
    1: ("Water", "Water"),
    2: ("Tree/Forest", "Tree/Forest"),
    3: ("Low vegetaion/field", "Low vegetaion/field"),
    4: ("Impervious", "Impervious")
}

CHESAPEAKE_4_NO_ZEROS_CLASS_DEFINITIONS = {
    0: ("Water", "Water"),
    1: ("Tree/Forest", "Tree/Forest"),
    2: ("Low vegetaion/field", "Low vegetaion/field"),
    3: ("Impervious", "Impervious")
}

CHESAPEAKE_7_CLASS_DEFINITIONS = {
    0: ("Unclassified", r"No data value"),
    1: ("Water", "Water"),
    2: ("Tree/Forest", "Tree/Forest"),
    3: ("Low vegetaion/field", "Low vegetaion/field"),
    4: ("Soil and Barren", "Soil and Barren"),
    5: ("Impervious (other)","Impervious (other)"),
    6: ("Impervious (road)","Impervious (road)"),
}

def get_colors(class_colors):
    return np.array([class_colors[c] for c in class_colors.keys()]) / 255.0
    
# denver functions read from utils
def create_map_raw_lc_to_idx(class_definitions):
    LC_TO_IDX = np.zeros(np.array(list(class_definitions.keys())).max() + 1, dtype=np.uint8)
    for i, k in enumerate(class_definitions.keys()):
        LC_TO_IDX[k] = i
    return LC_TO_IDX    
      
lc_colors = {
    'nlcd': get_colors(NLCD_CLASS_COLORS),
    'enviroatlas': get_colors(ENVIROATLAS_CLASS_COLORS),
    'chesapeake_7': get_colors(CHESAPEAKE_7_CLASS_COLORS),
    'chesapeake_5': get_colors(CHESAPEAKE_5_CLASS_COLORS),
    'chesapeake_4': get_colors(CHESAPEAKE_4_CLASS_COLORS),
    'chesapeake_4_no_zeros': get_colors(CHESAPEAKE_4_NO_ZEROS_CLASS_COLORS),
}

lc_cmaps = {}
for lc_type, colors in lc_colors.items():
    lc_cmaps[lc_type] = matplotlib.colors.ListedColormap(colors)
    
map_raw_lc_to_idx = {
    'nlcd': create_map_raw_lc_to_idx(NLCD_CLASS_DEFINITIONS),
    'enviroatlas': create_map_raw_lc_to_idx(ENVIROATLAS_CLASS_DEFINITIONS),
    'chesapeake_4_no_zeros': create_map_raw_lc_to_idx(CHESAPEAKE_4_NO_ZEROS_CLASS_DEFINITIONS),
}

class_definitions = {
    'nlcd': list(NLCD_CLASS_DEFINITIONS.values()),
    'enviroatlas': list(ENVIROATLAS_CLASS_DEFINITIONS.values()),
    'chesapeake_7': list(CHESAPEAKE_7_CLASS_DEFINITIONS.values()),
    'chesapeake_5': list(CHESAPEAKE_5_CLASS_DEFINITIONS.values()),
    'chesapeake_4': list(CHESAPEAKE_4_CLASS_DEFINITIONS.values()),
    'chesapeake_4_no_zeros': list(CHESAPEAKE_4_NO_ZEROS_CLASS_DEFINITIONS.values()),
}


# function taken from land-cover-private/notebooks/Figures - Dataset Legends
def make_legend_figure(lc_type):
    if lc_type == 'nlcd':
        class_color_dict = NLCD_CLASS_COLORS
        class_name_dict = NLCD_CLASS_DEFINITIONS
    elif lc_type == 'enviroatlas':
        class_color_dict = ENVIROATLAS_CLASS_COLORS
        class_name_dict = ENVIROATLAS_CLASS_DEFINITIONS
    elif lc_type == 'chesapeake_7':
        class_color_dict = CHESAPEAKE_7_CLASS_COLORS
        class_name_dict = CHESAPEAKE_7_CLASS_DEFINITIONS
    elif lc_type == 'chesapeake_5':
        class_color_dict = CHESAPEAKE_5_CLASS_COLORS
        class_name_dict = CHESAPEAKE_5_CLASS_DEFINITIONS
    elif lc_type == 'chesapeake_4':
        class_color_dict = CHESAPEAKE_4_CLASS_COLORS
        class_name_dict = CHESAPEAKE_4_CLASS_DEFINITIONS
    elif lc_type == 'chesapeake_4_no_zeros':
        class_color_dict = CHESAPEAKE_4_NO_ZEROS_CLASS_COLORS
        class_name_dict = CHESAPEAKE_4_NO_ZEROS_CLASS_DEFINITIONS
    else:
        print(f'landcover type {lc_type} not recognized')
    
    classes = class_name_dict.keys()
    class_names = [
        class_name_dict[class_name][0]
        for class_name in classes
    ]
    classes = class_color_dict.keys()
    patches = [
        matplotlib.patches.Patch(facecolor=np.array(class_color_dict[class_name])/255.0, edgecolor='k')
        for class_name in classes
    ]
    
    
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis("off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.legend(patches, class_names, loc='upper left', fontsize=17, frameon=False)
  #  plt.show()
   # plt.close()
    
    
def vis_lc(r, lc_type, renorm=True, reindexed=True):
    colors = lc_colors[lc_type]
    
    sparse = r.shape[0] != len(colors)
    colors_cycle  = range(0, len(colors))
   #    colors_cycle  = range(len(colors))
    if sparse:
        z = np.zeros((3,) + r.shape)
        s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * (s == c).astype(float)
        
    else:
        z = np.zeros((3,) + r.shape[1:])
        if renorm: s = r / r.sum(0)
        else: s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * s[c] 
    return z

def vis_lc_from_colors(r, colors, renorm=True, reindexed=True):
    sparse = r.shape[0] != len(colors)
    colors_cycle  = range(0, len(colors))

    if sparse:
        z = np.zeros((3,) + r.shape)
        s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * (s == c).astype(float)
        
    else:
        z = np.zeros((3,) + r.shape[1:])
        if renorm: s = r / r.sum(0)
        else: s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * s[c] 
    return z

def count_cooccurances(labels_1, labels_2, lc_type_1, lc_type_2):
    num_classes_1 = len(class_definitions[lc_type_1])
    num_classes_2 = len(class_definitions[lc_type_2])
    counts = np.zeros((num_classes_2, num_classes_1),dtype=int)

    # brute force way
    for c_1 in range(num_classes_1):
        for c_2 in range(num_classes_2):
            # compute co-occurances
            count = (labels_1[labels_2 == c_2] == c_1).sum()
            counts[c_2, c_1] = count

    return counts

