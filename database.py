class FakeDB:

    def save_review(self, pr_url):
        print(f"Saved {pr_url}")