# I wasn't happy with any of the GitHub libraries for Python that I tried so I
# just used the GitHub API directly.  If someone would like to rewrite this
# using a library please be my guest

import argparse
import base64
import getpass
import json
import logging
import sys
import urllib
import urllib2


BASE_URL = 'https://api.github.com/repos/'


log = logging.getLogger()


class _MaxLevelFilter(logging.Filter):
    def __init__(self, maxlevel):
        self.maxlevel = maxlevel

    def filter(self, record):
        return record.levelno <= self.maxlevel


class GithubRequestError(Exception):
    pass


class GithubSuggestBackports(object):
    # Cache all the commits found for the given branch so we don't have to
    # re-request them for each pull request
    _cached_commits = []

    def __init__(self, owner, repo, branch, username=None, password=None):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        if username is not None and password is not None:
            # We can't rely on urllib2 to handle basic authentication in the
            # normal way since GitHub requests don't always have
            # www-authenticate in the headers
            self._auth = base64.b64encode(':'.join((username, password)))
        else:
            self._auth = None

    def _github_repo_request(self, *resource, **parameters):
        url = BASE_URL + '/'.join((self.owner, self.repo) + resource)
        if parameters:
            url += '?' + urllib.urlencode(parameters)
        log.debug('Requesting ' + url)
        req = urllib2.Request(url)
        if self._auth:
            req.add_header('Authorization', 'Basic ' + self._auth)
        try:
            response = json.load(urllib2.urlopen(req))
        except urllib2.HTTPError, e:
            response = json.load(e.fp)
            if 'message' in response:
                raise GithubRequestError(response['message'])
            raise e
        return response

    def get_milestones(self, state=None):
        parameters = {}
        if state is not None:
            parameters['state'] = state

        return self._github_repo_request('milestones', **parameters)

    def get_issues(self, milestone=None, state=None):
        parameters = {}
        if milestone is not None:
            parameters['milestone'] = milestone
        if state is not None:
            parameters['state'] = state

        parameters['page'] = 1
        issues = []
        while True:
            response = self._github_repo_request('issues', **parameters)
            if response:
                issues.extend(response)
                parameters['page'] += 1
            else:
                break

        return issues


    def get_commits(self, sha):
        return self._github_repo_request('commits', sha=sha)

    def get_pull_request(self, number):
        try:
            pr = self._github_repo_request('pulls', str(number))
        except GithubRequestError, e:
            if e.message == 'Not Found':
                return None
            raise
        return pr

    def find_commit(self, sha, since=None):
        if not self._cached_commits:
            # Initialize with the first page of commits from the bug fix branch
            self._cached_commits = self.get_commits(self.branch)
        idx = 0
        while True:
            try:
                commit = self._cached_commits[idx]
            except IndexError:
                # Try growing the list of commits; but if there are no more to be
                # found return None
                last_commit = self._cached_commits[-1]
                next_commits = self.get_commits(last_commit['sha'])[1:]
                if next_commits:
                    self._cached_commits.extend(next_commits)
                    continue
                return None

            if commit['sha'] == sha:
                return commit

            if commit['commit']['author']['date'] < since:
                return None

            idx += 1

    def get_next_milestone(self):
        """Get the next open milestone that has the same version prefix as the
        branch.  For example if the repo has milestones v0.2.1 and v0.2.2 and the
        branch is v0.2.x, this will return v0.2.1.
        """

        prefix = self.branch[:-1]
        milestones = [m for m in self.get_milestones(state='open')
                      if m['title'].startswith(prefix)]
        sort_key = lambda m: int(m['title'].rsplit('.', 1)[1])
        return sorted(milestones, key=sort_key)[0]

    def iter_suggested_prs(self):
        next_milestone = self.get_next_milestone()
        log.info("Finding PRs in milestone {0} that haven't been merged into "
                 "{1}".format(next_milestone['title'], self.branch))
        for issue in self.get_issues(milestone=next_milestone['number'],
                                     state='closed'):
            pr = self.get_pull_request(issue['number'])
            if pr is None or not pr['merged']:
                continue
            import pdb; pdb.set_trace()
            sha = pr['merge_commit_sha']
            if self.find_commit(sha, since=pr['merged_at']):
                yield pr['number'], pr['title'], sha


def main(argv):
    parser = argparse.ArgumentParser(
        description='Find pull requests that need be backported to a bug fix '
                    'branch')
    parser.add_argument('owner', metavar='OWNER',
                        help='owner of the repository')
    parser.add_argument('repo', metavar='REPO', help='the repository name')
    parser.add_argument('branch', metavar='BRANCH',
                        help='the name of the bug fix branch (eg. v0.2.x)')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args(argv)

    # Configure log
    log.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    if args.debug:
        stdout_handler.setLevel(logging.DEBUG)
    else:
        stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    log.addHandler(stdout_handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    log.addHandler(stderr_handler)


    log.info("Enter your GitHub username and password so that API requests "
             "aren't as severely rate-limited...")
    username = raw_input('Username: ')
    password = getpass.getpass('Password: ')
    suggester = GithubSuggestBackports(args.owner, args.repo, args.branch,
                                       username, password)
    for num, title, sha in suggester.iter_suggested_prs():
        log.info('[#{0}] {1}: {2}'.format(num, title, sha))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
