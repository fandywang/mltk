#!/usr/bin/env  python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import struct

# Maximum record size, default = 64MB
SANITY_CHECK_BYTES = 64 * 1024 * 1024

class RecordWriter(object):
    """Write string records to a stream.

    Max record size is 64MB for the sake of sanity.
    """

    def __init__(self, fp):
        """Initialize a Writer from the file pointer fp.
        """
        self.fp = fp
        if ('w' not in self.fp.mode and
                'a' not in self.fp.mode and
                '+' not in self.fp.mode):
            logging.error("""Filehandle supplied to RecordWriter does not
                    appear to be writeable.""")

    def write(self, blob):
        """Append the blob to the current RecordWriter.

        Returns True on success, False on any filesystem failure.
        """
        if not isinstance(blob, str):
            logging.error('Invalid type, blob (type = %s) not StringType.'
                    % type(blob))
            return False

        blob_len = len(blob)
        global SANITY_CHECK_BYTES
        if blob_len > SANITY_CHECK_BYTES:
            logging.error('Record size %d exceeded.' % blob_len)
            return False

        self.fp.write(struct.pack('>L', blob_len))
        self.fp.write(blob)

        return True

class RecordReader(object):
    """Read string records from a RecordWriter stream.
    """

    def __init__(self, fp):
        """Initialize a Reader from the file pointer fp.
        """
        self.fp = fp
        if (('w' in self.fp.mode or 'a' in self.fp.mode) and
                '+' not in self.fp.mode):
            logging.error("""Filehandle supplied to RecordReader does not
                    appear to be readable.""")

    def read(self):
        """Read s single record from this stream.  Updates the file position
        on both success and failure (unless no data is available, in which case
        the file position is unchanged and None is returned.)

        Returns string blob or None if no data available.
        """
        blob = self.fp.read(4)
        if len(blob) == 0:
            logging.debug('%s has no data (current offset = %d).'
                    % (self.fp.name, self.fp.tell()))
            self.fp.seek(self.fp.tell())
            return None

        if len(blob) != 4:
            logging.error('Expected 4 bytes, but got %d.' % len(blob))
            return None

        blob_len = struct.unpack('>L', blob)[0]
        global SANITY_CHECK_BYTES
        if blob_len > SANITY_CHECK_BYTES:
            logging.error('Record size %d exceeded.' % blob_len)

        read_blob = self.fp.read(blob_len)
        if len(read_blob) != blob_len:
            logging.error('Premature end of stream.')
        return read_blob

