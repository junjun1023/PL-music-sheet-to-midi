"""
    Represents a single staff
    """
class Staff:
    #staff被呼叫是因為getting_lines最後return有呼叫到
    def __init__(self, min_range, max_range): #init類似於C++的constructer
        self.min_range = min_range
        self.max_range = max_range
        self.lines_location, self.lines_distance = self.get_lines_locations() #get_lines_location最後回傳兩個值就分別給location跟distance
    
    """
        Calculates the approximate positions of the separate lines in the staff
        :return: list of approximate positions of the lines
        """
    def get_lines_locations(self):
        lines = [] #empty list
        lines_distance = int((self.max_range - self.min_range) / 4)
        for i in range(5): #從0跑到4，5次有5條線
            lines.append(self.min_range + i * lines_distance) #append把每一次新的值放到最後的位子
        return lines, lines_distance
