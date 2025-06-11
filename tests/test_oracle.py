import oracledb
dsn = oracledb.makedsn('localhost', 1521, service_name='XEPDB1')
connection = oracledb.connect(user='sys', password='admin', dsn=dsn, mode=oracledb.SYSDBA)
print("Connection successful")
connection.close()