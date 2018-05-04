
class model_initializer:
  '''Add code here to detect the environment and set necessary variables before launching the model'''
  args=None
  custom_args=[]
  def __init__(self, args, custom_args=[]):
    self.args = args
    self.custom_args = custom_args
    if self.args.verbose: 
      print 'Received these standard args: {}'.format(self.args)
      print 'Received these custom args: {}'.format(self.custom_args)
      print 'Initialize here.'

  def run(self):
    if self.args.verbose: print "Run model here."
