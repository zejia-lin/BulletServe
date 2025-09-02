class auto:
    # Class variable to track all instances
    _instances = []
    
    # Class variable to track current class being defined
    _current_class = None
    
    def __init__(self):
        # Add this instance to the class registry
        auto._instances.append(self)
        self.value = None
    
    def __set_name__(self, owner, name):
        # When the class is created, this is called for each descriptor
        if auto._current_class != owner:
            # New class, reset counter
            auto._current_class = owner
            auto._counter = 0
        
        # Assign value and increment counter
        self.value = auto._counter
        auto._counter += 1
    
    def __get__(self, obj, objtype=None):
        return self.value
