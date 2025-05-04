import ee

def init_gee(project_name):

    # Trigger the authentication flow.
    ee.Authenticate()

    # Initialize the library.
    ee.Initialize(project = project_name)