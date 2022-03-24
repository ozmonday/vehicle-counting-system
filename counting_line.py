test_day = {
  'area-one' : {
    'tl' : (260, 200), #xy
    'tr' : (500, 200),
    'br' : (500, 530),
    'bl' : (260, 530)
  },
  'area-two' : {
    'tl' : (500, 200),
    'tr' : (600, 200),
    'br' : (600, 530),
    'bl' : (500, 530)
  },
  'area-three' : {
    'tl' : (),
    'tr' : (),
    'br' : (),
    'bl' : ()
  }
}

print(list(dict(test_day['area-one']).values()))