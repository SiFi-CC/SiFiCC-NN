class Fibre:
    """
    A class to represent a scintillating fibre.
    
    
    Attributes
    ----------
    position : list
        The position of the fibre's hit in 3d space.
        
    Methods
    -------
    __repr__():
        Returns a string representation of the Fibre object.
        
    Constructs all the necessary attributes for the Fibre object.
    
    Parameters
    ----------
    fibre : np.array
        A numpy array containing the position of the fibre's hit.
    """
    
    def __init__(self, fibre):
        self.position = fibre[:3]
    
    def __repr__(self):
        return f"Fibre(position={self.position})"