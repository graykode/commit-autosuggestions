Fix broken build due to @ RemotableViewMethod 
Added lightweight exception logging to AsyncQueryHandler . 
Adjust performance expectations 
Slide the whole window ( also fixes overlay AB issues ) . 
Add the checkstyle rule FinalClass 
Add precondition to throw more informative exception when calling request ( ) before start ( ) has been called . 
Hide ObjectGraph ' s constructor 
Fix failing test case 
decrease default cache size to prevent OOEMs 
remove duplicate setter 
Don ' t exit lock task on sleep 
fix test data 
Made field private 
ignore eclipse project files 
HHH - 5376 Tests contain invalid SQL 
make mesh static by default 
Help topic specified . 
Remove TODO 
For listObjects wit no prefix , insert a " / " as the path ( HTTP compliance ) 
Dismiss overflow popup when overflow button is detached . 
Add Google - internal memory consumption tests for all top - level Multiset implementations . 
Explicitly request the focus in the call panel in single window mode . 
removed unused import 
Updated todo 
Add a class PlainSaslServer that implements the SaslServer interface - - add newline at end of file 
removed whitespace 
removed debug printing 
enable cleaner hack , so it consumes less disk space 
Remove unnecessary isLeaderMode ( ) from MasterBase . java 
Fixed a bug where menu items from all fragments in pager would be added to menu 
update generated Order class 
Deprecate getAssociatedNode 
Fix JavaDoc . 
added system exit at the end of main call to avoid hanging problem in grails generate - all 
Peek the left overlay drawer . 
Remove hangover printStackTrace from closing UdpChannelTransports . 
changed the inner class to static 
Fixed EmptyValidator allowing whitespace 
Set INHERIT_LOG4J default to false . 
Removed system . out 
help id ( IDEA - 137826 ) 
Set unsubscribed true 
Add solution to problem 78 
Fix typo 
Enabling show implementations in plain html and other files that do not contain java code . 
Ignore MoreFilesIntegrationTest . 
Showing the dialog 
corrected check for duplicated classes 
GraphServiceBeanImpl always returns the same graph , even with routerId 
Added call to HystrixPlugins . reset ( ) in HystrixCommandTestWithCustomConcurrencyStrategy 
ignore failing test 
remove unused import 
Cannot edit watch in Flex debugger . 
Include 6 . 0 . x parameter processing fix in trunk . 
Fix tap detection after a long press 
DO NOT MERGE - - copy on tapping outside of selected text in gmail 
stopping the preview before unlocking reportedly helps with video corruption . . . 
Delete duplicate SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION flag 
Removed a System . out 
Fix uireader - ontranslate 
BubbleFlowDraggable handles the onDestroyBubble ( ) . 
Fixed compile error for the spelling pull request . 
fixed issue with detecting html tag state 
Eliminate an unnecessary initializer 
Hiding DrawerLayout does not fire correct accessibility event . 
missing import 
Add missing URI import 
Add ExchangeNode to graphviz printer 
Remove unused enum value 
Fix year picker initial range 
avoid massive logging 
Fix bug where SslHandler returned wrong value for getCloseNotifyTimeoutMillis ( ) 
Remove unnecessary @ SuppressWarnings . 
made class final 
Use the write listener to decide between a blocking and a non blocking flush on close . 
Fix sync tests failure . 
Removing dead store ; it was likely a relict of debuging code . 
added some logging to NodeEnvironmentTests . testDeleteSafe 
fixed wrong UnreachedBarColor # 25 
LPS - 52171 NewEnvTestRule . RunInNewClassLoaderStatement should reset MethodCache after run , to release references to the new classloader , so that it can be gced properly 
Adds FlakyTest annotation to failing acceptance test 
Check interface before invoking scheduleLowMemory 
id - based equals & hashCode generation 
Fix Issue # 550 
add missing method to mbean 
Remove outdated ES config 
Fix cleaning scan result cache 
When configuring mmapfs it is not used since 0 . 19 . 7 , closes # 2253 . 
Revert " LPS - 54998 SF " 
Fix for declaration resolve 
updated key types . 
Fix a typo in a javadoc comment 
added a private constuctor for ticket # TACHYON - 1243 
Reading / writing the mappings - - - tested version . 
Add missing case in switch . 
Removed superfluous @ SuppressWarnings ( " unused " ) from PutAllPartitionAwareOperationFactory 
Fix typo in exception . 
Assert statement formatting 
texture wrap enum 
do not hang tests on assertion 
Rename carrier config constant . 
added decent message when cusom matcher is used 
Fix NPE if there is no accelerometer . 
fixed missing import 
do not change detected platform with initPlatformPrefix 
Add RazorRepository to factory 
make APersistentVector . SubVector public 
Ensure only a single scroller animation callback per frame . 
Fixed Nullable / NotNull annotations . 
Fix a silly bug 
Remove http / wml from the DumpRenderTree skipped list , as this directory no longer exists . 
Set READ_REQUIRES_WRITE when initiating a handshake 
Don ' t override default SSL Socket Factory 
remove debug item 
Make go case sensitive language 
reduce visibility 
added missing copyright header 
Removed call to Thread . dumpStack ( ) . 
Removed println 
use parameterized logging 
added break to exit loop on first match 
Remove unneed clear in constructor . 
fix bug 
Fix HttpProxy . 
Fix server launching . 
Closes # 2786 , PixmapPacker # updateTextureAtlas returned prematurely . 
LPS - 49294 Fix compile error 
help ID ( PY - 4115 ) 
threadsName should be attribute in threads DSL 
memory leak fixed ( IDEADEV - 15230 ) 
marking data table as dirty after record deletion 
Disable early open compaction 
Fixed a subtle bug I happened to introduce last night . 
closes # 558 
Removed transition animation in audioplayer 
I did not fix activity check correctly last time . 
Force a WebView pictureset invalidation on screen rotation . 
LPS - 41762 Make admin have to agreedToTermsOfUse 
fixed StartVMTest . testNoAttach 
Do not return a super type if a class represents an interface . 
Remove selection highlight when replacing / deleting . 
Make the default folder name size in the folder list the same as the default account name size in the account list 
configure wizard to be modal or not 
LPS - 51081 Fix liferayHome default value 
Fix indents 
Remove NAME from attribute definitions 
Made Evaluator constructor public to allow custom implementations 
Removed @ Override on an interface for 1 . 5 compatibility . 
add PascalNameFilter 
Removed a System . out 
remove failed assertion 
Reverted var name change . 
LPS - 45500 - Missed property in rename 
test refactored cont 
Add a package private setTypeRegistry ( ) function to Compiler 
Remove erroneous JavaDoc 
Fix AppCompatCheckedTextView tinting compound drawables 
removed unnecessary field assignment 
Really leave text selection mode when using the trackball . 
Removed unnecessary import . 
added finally block that sets the system property back to null after use so as not to break other tests 
getFileName ( ) should return the provided filename instead of a constant one 
Fix layer drawable inflation 
commit check 
Fix typo 
Enabled test of constructor signature there are unresolved superclasses ( PY - 4419 ) 
fix ` setDescription ` does not set the value correctly 
fixed TestMusicPlayer to corectly initialize audioContext 
record missing metric 
Fix the build 
Prefer Integer . valueOf over explicit boxing 
commit transaction after indexing . 
Fix tabs 
Fixed the fade color initialization 
Rolling back ven ' s optimization . 
Set database version to 12 . 
Fix typo 
IDEA - 84892 File annotations do not update on git commit 
Fix YAML Language display name 
invoke connection . close method to close es connection 
Do not return subscriber connection to prevent getting events 
which will give us a reference to the Object , that ' s probably not what we want 
Removed an unused variable . 
Organize import . 
jsvc actually wants the method to be named init , not load ( contrary to what their docs state in at least one place ) 
added @ Deprecated annotation to deprecated methods . 
Fix poi types 
Add missing @ XmlAttribute in model 
Fixed race condition in DBTCPConnector . authenticate by putting in a call to checkMaster prior to getting a DBPort to authenticate on . 
consolify submitted text when rendering incrementally 
Record the dsym . 
add default ctor to Lexer 
Fix deadlock 
Fix missing activate boolean 
help ID ( PY - 4114 ) 
Fixed failing formatter tests . 
Remove override annotation so we can use the latest SDK to compile 
Fix database upgrade 
LPS - 7524 Forcibly kill DBBuilder process after building database 
Make TranslucentConversionListener a SystemApi 
Ignoring test that is redundant for com . zaxxer . hikari . pool . TestConnectionTimeoutRetry . testConnectionRetries ( # 678 ) 
Make an internal class package private 
Fix logic error in CarrierService . 
LPS - 55175 Fix SF 
Corrected method name setSelectedNavigationIndex to setSelectedNavigationItem in the exception message 
added helper method for builder implementations 
HHH - 6421 Reduce log level of LazyInitializationException from ERROR to TRACE 
Declare two fields as final in BsonBinaryReader . Mark 
Fix isUsbMassStorageConnected ( ) will always return true 
Add some g3docs for the old and new type systems . 
between works only with numbers , should also support dates 
Fixed a bug where the expand button was invisible 
suppress warning 
added highlightinglexertest to AllTestsSuite 
turned off domain name verification when " verifiy " checkbox is unchecked 
clock skew detection needs to be closable , it doesn ' t fix itself 
added PLoggerFactory . unregister ( ) 
Properly set back SSLParameters , see # 810 
Add missing addPauseListener . 
Add Protocol . BASE_ETHERNET 
Don ' t play lock sound when just resetting Keyguard 
return the correct format for Instruction35s 
Fix some test failures 
increasing ROUND_JOIN_THRESHOLD in LineStroker to allow for line 
Restored the nonResizable nature of Font Size Field 
Removing wrong equals method in comparator . 
Call setExtent ( ) when setWorld ( ) is caleld on ParserContext . 
added constructor that takes the parent class loader 
fixed checkstyle error 
added missing moco runner 
Fix copy / paste error - thanks kkolinko 
restore completion settings ! 
Add missing used flag for write ( int ) . 
remove dangerous CTOR - not used in this driver 
Remove a random blank line . 
Make git status command silent 
Drop unused private member mIgnoreGutter in ViewPager 
Interrupt the current thread on catching InterruptedExcepted in the test runner . 
enable copy reference in modal context ( IDEA - 13942 ) 
Fix Spinner not restoring selection when set to GONE 
Add javadoc 
Remove debug code that was causing console spam in Forge . 
moved getGraph to graph 
Throw an exception if unsupported verb is used 
Remove unused code 
Fix SupervisorProperty display name . 
turn off indeterminate progress thing on sign out 
Updated version in Mongo class 
IDEA - 53534 Fix wrong configurable selection on settings reopen 
Increased number of items for render from 5 to 10 
Fix GTK + background in JBTable 
removed warning 
remove un - warning 
don ' t suggest java class names in text fields with autocompletion ( IDEA - 117213 ) 
Added better nested trxn tests . 
Fix bug with adding search source . 
Fix build due to added API 
update OK button 
Remove unnecessary import in DisplayUtils 
Fix missing break in Icon class 
undo a change 
Fix typo , thanks to Benjamin Hanzelmann 
Renamed " parent " to " endSubTable " method . 
set filename 
Fix always defaulting to DiskCacheAdapter 
remove unused field 
getEntries ( ) doesn ' t really throw exceptions 
Fix broken trust agent reloading v2 
Deactivate PDF export . 
added release notes for 3 . 2 . 2 patch 
Add missing ` @ Override ` annotation . 
add todo 
CompositeShader , add unimplemented method to fix warning 
Fix an IllegalArgumentException for the parent frame of a file dialog . 
remove spurious import 
Fix incorrect merge conflict to klp - modular - dev - plus - aosp 
include cause of configuration exception that Shouldn ' t Happen 
Fixed compile error caused by final keyword 
Fixed crash when playbackservice wasn ' t running and no last played media 
Updated schema version 
Remove unused import . 
Fix the build when ViewDebug . TRACE_RECYCLER is set to true . 
Don ' t reference the enclosing type instance when it ' s not used ( memory optimisation ) 
Removed a notification statement from the method ExtensionHistory # addHistory ( HistoryReference ) as the list was already notified when that statement is reached . 
change result type to long ( copy & paste error ) 
Add @ SuppressWarnings ( " serial " ) to an Exception . 
Set activity state correctly when ending activity . 
Add missing R file change . 
Fix a bug that findbugs found 
Fix ping 
Fix constructor error introduced in last checkin 
Change the scope 
Remove unused import in Scheduler . java 
add pm 
Add the rather crucial missing i + + in the wait loop 
Remove unnecessary rule key inclusion . 
Adding missing break statement in VideoCallProvider . 
Removed bogus comment 
Add channels to the schema . 
Reduced the radio log 
prune zero count rankings after each emit 
LPS - 51886 add @ override 
Fix reading sleep policy 
Removed unused import in Format class 
Do not expose WifiManager . isNanSupported as new API 
Addressing review comments . 
change JdbcDate to return Types . TIMESTAMP 
Fix compile error 
IDEA - 115227 No auto completion for non getter jsp expression language methods 
remove unhelpful System . out . println 
add TODO in CBA rebalancing 
remove debug message 
Removed close ( ) override that did nothing and did not call super . close ( ) 
fixing version number 
update color 
Set max pool size before core to avoid IAE by OpenJDK implementation . 
cleanup in pde properties , turn off lighting by default in ProcessingApplet 
Added override tag 
Fixed merge mistake . 
finish the same action that was started ; cleanup and generify 
Removed printStackTrace 
change version num and year for the release of libsvm 3 . 15 
Added generic argument to ICollection 
exclude incompatible test case 
Optimize imports . 
Hide the keyguard scrim when the status bar comes back . 
LPS - 39399 We shouldn ' t return - 1 
Add the inadvertently missing SCHEME string value provider 
Breaks the loop 
Increase NamedHealthCheck # getName visibility to public 
removed non - functional history tree provider 
Made getDriverClassName ( ) be protected and not part of the interface . 
Working out the kinks 
Removing @ Override to fix the build 
Revert " marked class as abstract " 
removed spurious System . out . println ( ) 
make Flusher a static inner class 
Fix transaction ( ) returning null 
Ease the validations on DatabaseConfiguration # url . 
Remove dead store to environmentConfig . 
Remove unused import . 
do not touch linefeeds 
Updated version 
Fix LayoutInflater copy constructor logic . 
Prevent TypeMap ' 0 ' being useful 
Fix checkstyle errors 
Add a class PlainSaslServer that implements the SaslServer interface - - fix the method name from setup to before 
Revert " Remove spurious @ Override . " 
Fix compile - time issue with generics 
Fix source formatter 
Added IPC exception ( support info ) 
fixed javadoc 
remove unused field 
Deprecate also implementation method 
remove empty static initializer from unit test for on master branch 
Add documentation for size ( ) 
LPS - 22629 Fix bug where verifyUrlTitle was accidentally removed in 6f3f31f5d747991cf4acba93307eccfbfbf824d3 
Spell better 
Fix issue with arrow not enable when setting maxDate 
Keep the debug message for now 
Fix typo . 
Fix deselction of all gpx 
Fixed formatting of data structure test code 
Make OreDictionary . initVanillaEntries ( ) private so that dumb modders will stop calling it . 
Fix issue with ForwardingEjbClientConfiguration 
LPS - 64771 Set _latestHASH before _createPluginsSummary ( ) ; 
LPS - 11221 Source formatting ( added a line break ) 
Return the document if a < / stream > is received . 
fixed wrong suite 
build number 11 
Clears reference to certificate dialog service on bundle stop . 
Allow JSC to understand the basic class relationships ( excluding interfaces ) . 
Empty Line Separator Check compilable UT inputs 
Fix rare segfault with proper size ( ) and check for null ids 
dispatch up event to fix fling 
Use correct version code for summer time fix upgrade . 
Removed spurious log about blinking cursor 
Cache recordsStores should be cleaned up during service shutdown 
ignore this for now 
LPS - 61129 Create inner exception class 
remove unused found from GHResponse 
Add getBytes to InputStream , URL and File ( patch thanks to jeremi Joslin ) 
fixed checkstyle error 
prune empty directories when switching to just created tag or branch 
fixed NPE 
Added missing test flag 
restore initial selection in NameSuggestionsField ( IDEADEV - 20506 ) 
Check mWebViewCore to prevent potential NPE . 
Remove checked exception from Buffer . readByteArray . 
restored @ Test annotation in SimpleValidateQueryTests 
Don ' t log interruption here , as it happens and it ' s ok ( EA - 36726 ) . 
Lowering recon batch size from 7 to 3 to avoid timeout problem . 
Pass environment from settings to console process . 
Remove redundant static modifier 
Added missing " final " keyword 
fix the robot smoke particle causing fps drops in certain cases 
Remove unused import 
Remove public from ScriptIntrinsicYuvToRGBThunker definition . 
data type convert error . 
LPS - 59594 Remove unused variable 
Fix an erroneous import 
Remove unneeded method 
Allow null filenamePattern in RequestLogConfiguration 
after project share performing some actions remain disabled until project reopening ) 
don ' t touch platform prefix if app is already initialized 
Implemented pre - key lookup 
Fixed typo in doc 
fixed failing master 
set a REASON_CHANGE_REQUESTED_BY_USER code to the unregister event when caused by us . 
Fix thread interrupt test 
Restored public access for Entities . EscapeMode 
Removes an empty line 
removed the commented - out method per discussion . 
Remove obsolete comment 
Added a property to avoid the exception stack trace when running the console 
Have output for valid configuration . 
CORE - 1661 Add support for MSSQL sequences 
Delegate exclude mimetypes to FasttrackBadgeWidget 
Added missing @ Override annotation in QueryCacheRecord . 
Try 8 KiB segments . 
deprecate unused method 
Fix hillshade layer appearance 
Fix error in settings . db upgrader . 
Remove backup of sleep policy setting 
always focus ok button in message dialog 
Don ' t change keyboard state in setIconifiedByDefault . 
Added a protected method to retrieve the indent printer for reuse in subclasses of MarkupBuilder 
Fix @ since for EscaperAsserts . 
Fix typo 
Investigating BZ 59261 
Fixed infinite loop in FileObserverService 
Actually return the registry entry , not null . 
Updated Timeline to handle fetchack message type . 
Fix build break 
LPS - 55026 add @ Deprecated 
make minute standard histogram interval . 
Fix a bug in Android Keystore updateAAD . 
Finish any enter animation when jumping to state . 
Revert r . 831830 . 
Check AirplaneMode when enabling Bluetooth has finished . 
remove unusage service type . 
Make ControlFlowGraph public 
Hiding soft keyboard on site selection 
Added assertion message 
Suppressing deprecation warning in a test 
reduce visibility 
Fix checkstyle 
fixed issue with implementation of getFiles in UsageTarget 
Allow class balance options without expert_mode . 
removed the get name method as operation set class names are going to be used for referencing them and not custom names . 
