"""Selected-event export must exclude process/environment metadata."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from nsight_trace import export_events, read_events


class NsightExportTest(unittest.TestCase):
    def test_export_keeps_only_referenced_event_strings_and_omits_environment_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            with closing(sqlite3.connect(p/'trace.sqlite')) as db:
                db.executescript('''
                    CREATE TABLE StringIds (id INTEGER, value TEXT);
                    INSERT INTO StringIds VALUES (1,'cudaGraphLaunch'),(99,'PRIVATE_ENV=example');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1);
                    CREATE TABLE NVTX_EVENTS (text TEXT, textId INTEGER);
                    INSERT INTO NVTX_EVENTS VALUES ('request/b64_s64/graph_open_once/0/0',NULL),('unrelated metadata',NULL);
                    CREATE TABLE TARGET_INFO_SYSTEM_ENV (value TEXT);
                    INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES ('PRIVATE_ENV=example');
                ''')
                db.commit()
            result = export_events(p)
            tables, strings = read_events(p/'events.sqlite')
            self.assertEqual(strings, {1: 'cudaGraphLaunch'})
            self.assertNotIn('TARGET_INFO_SYSTEM_ENV', tables)
            self.assertEqual(len(tables['NVTX_EVENTS']), 1)
            self.assertEqual(result['counts']['StringIds'], 1)
            with self.assertRaises(ValueError):
                export_events(p)

    def test_rejects_extra_metadata_table_in_published_event_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'events.sqlite'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE TARGET_INFO_SYSTEM_ENV (value TEXT)')
                db.commit()
            with self.assertRaises(ValueError):
                read_events(path)
