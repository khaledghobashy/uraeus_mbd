
# Standard library imports
import os
import shutil

###############################################################################

def create_standalone_project(project_name, parent_dir=''):
    path = os.path.abspath(parent_dir)
    project_path = os.path.join(path, project_name)
    
    shutil.copytree('_template_structure/standalone_project', project_path)
    
    os.chdir(project_path)
    cwdir = os.path.abspath(project_path)
    print('Current working directory is %s'%cwdir)


project_name = input('Please enter the "project_name": ')

create_standalone_project(project_name, 'standalone_models')

