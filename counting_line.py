test_day_1 = {
  'area-one' : {
    'tl' : (0.28, 0.38), #xy
    'tr' : (0.52, 0.38),
    'br' : (0.52, 0.98),
    'bl' : (0.28, 0.98)
  },
  'vehicles' : {
    'heavy-vehicle' : 1,
    'light-vehicle' : 25,
    'motor-vehicle' : 40,
    'unknow-vehicle' : 1
  }
  
}

test_day_2 = {
  'area-one' : {
    'tl' : (0.72, 0.40), #xy
    'tr' : (0.83, 0.49),
    'br' : (0.49, 0.89),
    'bl' : (0.33, 0.69)
  },
  'vehicles' : {
    'heavy-vehicle' : 1,
    'light-vehicle' : 25,
    'motor-vehicle' : 40,
    'unknow-vehicle' : 1
  }
  
}

test_day_3 = {
  'area-one' : {
    'tl' : (0.11, 0.6), #xy
    'tr' : (0.37, 0.61),
    'br' : (0.65, 0.65),
    'bl' : (0.89, 0.72),
    'ub' : (0.86, 0.88),
    'ut' : (0.63, 0.8),
    'uu' : (0.36, 0.76),
    'us' : (0.04, 0.74)
  },

  'vehicles' : {
    'heavy-vehicle' : 1,
    'light-vehicle' : 25,
    'motor-vehicle' : 40,
    'unknow-vehicle' : 1
  }
  
}

test_day_4 = {
  'area-one' : {
    'tl' : (0.72, 0.40), #xy
    'tr' : (0.83, 0.40),
    'br' : (0.70, 1),
    'bl' : (0.55, 1)
  },
  'vehicles' : {
    'heavy-vehicle' : 1,
    'light-vehicle' : 25,
    'motor-vehicle' : 40,
    'unknow-vehicle' : 1
  }
  
}


test_night_5 = {
  'area-one' : {
    'tl' : (0.72, 0.40), #xy
    'tr' : (0.83, 0.40),
    'br' : (0.70, 1),
    'bl' : (0.55, 1)
  },
  'vehicles' : {
    'heavy-vehicle' : 1,
    'light-vehicle' : 25,
    'motor-vehicle' : 40,
    'unknow-vehicle' : 1
  }
  
}





print(list(dict(test_day_1['area-one']).values()))