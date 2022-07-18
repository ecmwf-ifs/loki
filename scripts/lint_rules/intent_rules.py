from loki.lint import GenericRule, RuleType

class IntentCheckRule(GenericRule):  # Coding standards x.x

    type = RuleType.ERROR

    docs = {
        'id': 'x.x',
        'title': ('Rules for declared argument intent: '
                  'Check if use of variable matches declared intent'),
    }

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config):
        pass

# Create the __all__ property of the module to contain only the rule names
__all__ = tuple(name for name in dir() if name.endswith('Rule') and name != 'GenericRule')
