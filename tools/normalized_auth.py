"""Explicit policy-feature projection; arithmetic is done by code, not the LLM."""
import json
from make_specialist_data import SYSTEM


def features(record):
    return {'has_minimum_failed_logins':record['failed_logins']>=5,
            'within_policy_window':record['window_seconds']<=300,
            'subsequent_success':record['subsequent_success']}


def rule_label(record):
    facts=features(record)
    if not facts['has_minimum_failed_logins'] or not facts['within_policy_window']:
        return 'low'
    return 'high' if facts['subsequent_success'] else 'medium'


def messages(record,language,split='test'):
    rule_en=('Rule: if has_minimum_failed_logins is false OR within_policy_window is false, return low. '
             'Otherwise, return high if subsequent_success is true, and medium if it is false.')
    rule_ar=('القاعدة: إذا كان has_minimum_failed_logins يساوي false أو within_policy_window يساوي false فأعد low. '
             'وإلا فأعد high عندما يكون subsequent_success يساوي true، وmedium عندما يساوي false.')
    surface={'train':'Classify these validated Boolean facts / صنّف الحقائق المنطقية:\n',
             'valid':'Assign the priority from these facts / حدد الأولوية من الحقائق:\n',
             'test':'Audit this projected record / راجع السجل بعد الإسقاط:\n'}[split]
    return [{'role':'system','content':SYSTEM},
            {'role':'user','content':(rule_ar if language=='ar' else rule_en)+'\n'+surface+json.dumps(features(record))}]
