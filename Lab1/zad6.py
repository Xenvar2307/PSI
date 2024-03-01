class Article:
    def __init__(self, Title, Authors, Body, Year) -> None:
        self.Title = Title
        self.Authors = Authors
        self.Body = Body
        self.Year = Year

    def printInfo(self):
        print(f"Title: {self.Title}")
        print("Authors:")
        for author in self.Authors:
            print(author)
        print(f"Published year: {self.Year}")
        print(self.Body)

    def addAuthor(self, author):
        self.Authors.append(author)

    def changeTitle(self, newTitle):
        self.Title = newTitle

    def getBodyLength(self) -> int:
        return len(self.Body)

    def howOld(self, currentYear) -> int:
        return currentYear - self.Year


test = Article(
    "Tytuł testowy", ["Jan Nowak", "Agata Wiśniowska"], "Lorem Ipsum...", 1999
)
test.printInfo()
test.addAuthor("ja")
test.printInfo()
