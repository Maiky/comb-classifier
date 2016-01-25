
class ImageEntry:

    def __init__(self, camera_id, year, month, days, hours, minutes, seconds, microseconds, filename):
        self.camera_id = camera_id
        self.year = year
        self.month = month
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.microseconds = microseconds
        self.filename = filename
        self.origsize = None